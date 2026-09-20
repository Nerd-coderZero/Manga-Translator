import os
import sys

BACKEND_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "backend")
sys.path.insert(0, BACKEND_DIR)

from pipeline_core import is_garbage_text

_results = []


def check(name, condition, detail=""):
    _results.append((name, bool(condition), detail))
    mark = "PASS" if condition else "FAIL"
    print(f"[{mark}] {name}" + (f"  --  {detail}" if detail else ""))
    return bool(condition)


def run_checks():
    # reported gap: short misreads below the length-6/8 floors on the
    # repeated-character and digit-heavy checks used to pass through
    # unflagged. these are the two examples the gap was reported against.
    check("'CSB' (all-consonant misread) is flagged", is_garbage_text("CSB") is True)
    check("'d00' (letter/digit misread) is flagged", is_garbage_text("d00") is True)

    # same two failure shapes, different characters, to confirm the fix is
    # a general rule and not a hardcode against the two reported strings.
    check("'XKQ' (all-consonant) is flagged", is_garbage_text("XKQ") is True)
    check("'3d0' (digit-majority letter/digit mix) is flagged", is_garbage_text("3d0") is True)
    check("'9x2' (digit-majority letter/digit mix) is flagged", is_garbage_text("9x2") is True)

    # common short real words and abbreviations must still pass. this is
    # not an exhaustive real-world sample -- it is the reviewer's own set,
    # not drawn from actual OCR output -- so it demonstrates the fix is not
    # trivially over-broad, not that it is free of false positives on real
    # pipeline data.
    legit = ["OK", "Hi", "Mom", "Sir", "Run", "Wow", "Yes", "No", "TV", "Mr",
             "Dr", "PC", "SOS", "42", "7", "ID", "US", "UK", "Ms", "Go",
             "Ah", "Oh", "Ha", "No!", "Hey", "Now", "Wait"]
    for word in legit:
        check(f"{word!r} is not flagged", is_garbage_text(word) is False)

    # documented, accepted false positive: a short token that is mostly
    # digits with exactly one letter reads as OCR noise under this rule
    # even when it is real content. see the comment in pipeline_core.py
    # and docs/BUGS.md.
    check(
        "'3D' is a known accepted false positive (documented, not fixed)",
        is_garbage_text("3D") is True,
        "digit-majority letter/digit mix rule catches real alphanumeric tokens this short too",
    )

    # documented, accepted remaining gap: a digit-minority letter/digit mix
    # is deliberately not caught, to avoid the false-positive risk above.
    # this is the inverse tradeoff of the '3D' case.
    check(
        "'O0O' remains unflagged (known accepted gap, not fixed)",
        is_garbage_text("O0O") is False,
        "digit-minority letter/digit mixes are left alone to avoid flagging tokens like '3D'",
    )

    # the pre-existing longer-string checks (repeated-character dominance
    # at length >= 8, digit-majority at length >= 6) must be unchanged --
    # the fix only adds a new branch for length <= 5 and does not touch them.
    check("long repeated-character string still flagged", is_garbage_text("aaaaaaaa") is True)
    check("long digit-majority string still flagged", is_garbage_text("123456") is True)
    check("ordinary sentence-length text still passes", is_garbage_text("店長ちょっと待って") is False)

    # unchanged edge cases from before the fix.
    check("empty string is flagged", is_garbage_text("") is True)
    check("whitespace-only string is flagged", is_garbage_text("   ") is True)
    check(
        "string over max_reasonable_length is flagged",
        is_garbage_text("a" * 81) is True,
    )

    # check_length=False (pipeline.py's post-merge call): a long but real,
    # merge-produced block of Chinese dialogue must not be dropped just for
    # being long. this reproduces the length-cap bug found via a real
    # PaddleOCR merge of Chinese dialogue that exceeded the 80-char cap.
    long_real_dialogue = (
        "你不要再欺负她了，她已经很可怜了，你还要这样对她，"
        "我真的看不下去了，求求你放过她吧，我求你了，"
        "这样对一个女孩子实在是太过分了，希望你能够明白这一点，"
        "拜托你了，我们都是朋友，不要再这样下去了"
    )
    assert len(long_real_dialogue) > 80, (
        f"test string must exceed the 80-char cap to be meaningful, got {len(long_real_dialogue)}"
    )
    check(
        "long real merged dialogue is NOT flagged when check_length=False",
        is_garbage_text(long_real_dialogue, check_length=False) is False,
    )
    check(
        "the same text IS flagged when check_length=True (default)",
        is_garbage_text(long_real_dialogue) is True,
    )

    # reported gap: real manga dialogue relying on repeated punctuation for
    # stammering ("...") or emphasis ("！！！", "~~") was being dropped by
    # the repeated-character-dominance check, because punctuation counted
    # toward the dominance ratio. these are real strings from a 90-page
    # bulk run's "still empty" regions (source text only, not reproduced
    # from any copyrighted work -- this is the manga's own OCR'd dialogue).
    real_empties_now_rescued = [
        "不会...让妓女大人...白踩着我排泄的...我...我会付钱的...唔唔...",
        "是..我....这...这是这次的钱....",
        "喝~喝~~喝~喝~~",
        "!!!??!..妈..妈...",
        "嗯...好...好",
    ]
    for s in real_empties_now_rescued:
        check(
            f"real dialogue {s[:20]!r}... no longer flagged by punctuation-dominance",
            is_garbage_text(s, check_length=False) is False,
        )

    # documented, accepted remaining gap: a genuinely repeated semantic
    # (non-punctuation) character -- real onomatopoeia/mockery in this
    # manga's dialogue -- is structurally identical to OCR noise repeating
    # one glyph and is still flagged. not fixed this pass; see the comment
    # in pipeline_core.py and docs/BUGS.md.
    check(
        "'他女儿~哼哼哼哼哼哼哼~谁知道呢' (repeated real syllable) still flagged (accepted gap)",
        is_garbage_text("他女儿~哼哼哼哼哼哼哼~谁知道呢", check_length=False) is True,
    )

    # the short-token checks (<=5 chars) are unaffected by the punctuation
    # exclusion and continue to catch genuine short OCR misreads.
    check("'达4' (real reported OCR misread) is still flagged", is_garbage_text("达4", check_length=False) is True)

    total = len(_results)
    passed = sum(1 for _, ok, _ in _results if ok)
    print(f"\n{passed}/{total} checks passed")
    return passed == total


if __name__ == "__main__":
    ok = run_checks()
    sys.exit(0 if ok else 1)
