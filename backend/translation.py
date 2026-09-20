import os
import re
import time
import requests

NVIDIA_API_KEY = os.environ.get("NVIDIA_API_KEY")
NVIDIA_CHAT_URL = "https://integrate.api.nvidia.com/v1/chat/completions"

# Which translator to use:
#   auto      Nemotron first, then the free fallback (Google, then MyMemory).
#             This is the default and what bulk runs should use.
#   nemotron  Nemotron only, no fallback (old behaviour).
#   fast      skip Nemotron entirely, go straight to the free fallback. Quickest
#             option for very large batches.
TRANSLATION_PROVIDER = os.environ.get("TRANSLATION_PROVIDER", "auto").strip().lower()

# Nemotron gets very few retries in auto mode: a fallback exists, so waiting
# minutes on a flaky line is worse than just handing it to the fast translator.
NEMOTRON_MAX_RETRIES = int(os.environ.get("NEMOTRON_MAX_RETRIES", "2"))
NEMOTRON_TIMEOUT_SECONDS = int(os.environ.get("NEMOTRON_TIMEOUT_SECONDS", "30"))

_CJK_RE = re.compile(r"[぀-ヿ㐀-䶿一-鿿ｦ-ﾟ]")

MANGA_SYSTEM_PROMPTS = {
    "zh": (
        "You are translating dialogue from a Chinese manga/comic into English. "
        "Lines are short, casual spoken dialogue between characters, not formal writing. "
        "Keep the translation short and natural, matching how people actually speak. "
        "Preserve tone (angry, surprised, teasing, etc.) and keep any sound effects or "
        "exclamations punchy rather than literal. Output only the translated line, "
        "with no explanation, no notes, and no quotation marks around it."
    ),
    "ja": (
        "You are translating dialogue from a Japanese manga/comic into English. "
        "Lines are short, casual spoken dialogue between characters, not formal writing. "
        "Keep the translation short and natural, matching how people actually speak. "
        "Preserve tone (angry, surprised, teasing, etc.) and keep any sound effects or "
        "exclamations punchy rather than literal. Pay attention to sentence-final particles "
        "and honorifics for tone, but do not translate honorifics literally into English titles "
        "unless natural. Output only the translated line, with no explanation, no notes, "
        "and no quotation marks around it."
    ),
}


def translate_text_nemotron(text, source_lang="zh"):
    if not NVIDIA_API_KEY:
        raise RuntimeError("NVIDIA_API_KEY environment variable is not set")

    system_prompt = MANGA_SYSTEM_PROMPTS.get(source_lang, MANGA_SYSTEM_PROMPTS["zh"])

    headers = {
        "Authorization": f"Bearer {NVIDIA_API_KEY}",
        "Accept": "application/json",
        "Content-Type": "application/json",
    }
    payload = {
        "model": "nvidia/nemotron-3-super-120b-a12b",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text},
        ],
        # thinking is off: it added seconds per line and, with a token cap,
        # could burn the whole budget on reasoning and return empty content.
        # a short line needs a short answer, so max_tokens is small too.
        "temperature": 0.3,
        "top_p": 0.9,
        "max_tokens": 512,
        "stream": False,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    response = requests.post(
        NVIDIA_CHAT_URL, headers=headers, json=payload, timeout=NEMOTRON_TIMEOUT_SECONDS
    )
    response.raise_for_status()
    result = response.json()
    message = result["choices"][0]["message"]

    # content can be null on NIM when the model produced nothing usable
    return (message.get("content") or "").strip()


def _is_valid(translated):
    # a "translation" that is empty or still contains Japanese/Chinese
    # characters means the model skipped or echoed the line. treat it as a
    # miss so the fallback translator gets a turn.
    if not translated or not translated.strip():
        return False
    if _CJK_RE.search(translated):
        return False
    return True


def translate_text_google(text, source_lang="zh"):
    # deep-translator scrapes Google Translate's free endpoint: no key needed
    from deep_translator import GoogleTranslator

    source = "ja" if source_lang == "ja" else "zh-CN"
    return (GoogleTranslator(source=source, target="en").translate(text) or "").strip()


def translate_text_mymemory(text, source_lang="zh"):
    # second free library, also keyless (deep-translator's MyMemory wrapper)
    from deep_translator import MyMemoryTranslator

    source = "japanese" if source_lang == "ja" else "chinese simplified"
    # an email (any address, no signup) raises MyMemory's free daily quota
    kwargs = {"email": os.environ["MYMEMORY_EMAIL"]} if os.environ.get("MYMEMORY_EMAIL") else {}
    return (MyMemoryTranslator(source=source, target="english", **kwargs).translate(text) or "").strip()


FAST_PROVIDERS = [
    ("google", translate_text_google),
    ("mymemory", translate_text_mymemory),
]

# Google's free endpoint allows about 5 requests/second and Kaggle/Colab share
# IPs, so it often answers "too many requests". when a provider says that, it
# is skipped for a while instead of being retried on every single line.
COOLDOWN_SECONDS = 60
MIN_GAP_SECONDS = {"google": 0.35}
_cooldown_until = {}
_last_call = {}


def _is_rate_limited(exc):
    text = f"{type(exc).__name__} {exc}".lower()
    return "toomanyrequests" in text or "too many" in text or "429" in text


def _translate_fast(text, source_lang):
    """Free libraries, no API keys: Google first, MyMemory if Google fails."""
    now = time.time()
    available = [p for p in FAST_PROVIDERS if now >= _cooldown_until.get(p[0], 0)]
    # if every provider is cooling down, try them all anyway rather than give up
    for name, fn in (available or FAST_PROVIDERS):
        for attempt in range(2):
            gap = MIN_GAP_SECONDS.get(name, 0)
            wait = _last_call.get(name, 0) + gap - time.time()
            if wait > 0:
                time.sleep(wait)
            _last_call[name] = time.time()
            try:
                translated = fn(text, source_lang)
                if _is_valid(translated):
                    print(f"[{name}] translated ok ({len(text)} -> {len(translated)} chars)")
                    return translated
                print(f"[{name}] returned unusable output for {len(text)}-char source")
                break
            except Exception as e:
                print(f"[{name}] error ({len(text)}-char source, attempt {attempt + 1}/2): "
                      f"{type(e).__name__}: {str(e)[:100]}")
                if _is_rate_limited(e):
                    _cooldown_until[name] = time.time() + COOLDOWN_SECONDS
                    print(f"[{name}] rate limited, skipping it for {COOLDOWN_SECONDS}s")
                    break
                time.sleep(1)
    return ""


def _translate_nemotron_with_retries(text, source_lang, max_retries, base_wait_seconds):
    for attempt in range(max_retries):
        try:
            translated = translate_text_nemotron(text, source_lang)
            if _is_valid(translated):
                print(f"[nemotron] translated ok ({len(text)} chars source -> {len(translated)} chars output)")
                return translated
            print(f"[nemotron] unusable output for {len(text)}-char source "
                  f"(attempt {attempt + 1}/{max_retries})")
        except Exception as e:
            print(f"Nemotron error ({len(text)}-char source, attempt {attempt + 1}/{max_retries}): {e}")

        if attempt < max_retries - 1:
            time.sleep(base_wait_seconds * (2 ** attempt))
    return ""


def translate_text(text, source_lang="zh", max_retries=None, base_wait_seconds=2):
    provider = TRANSLATION_PROVIDER

    if provider == "fast" or (provider == "auto" and not NVIDIA_API_KEY):
        return _translate_fast(text, source_lang)

    if max_retries is None:
        # without a fallback, keep the old patient behaviour
        max_retries = 5 if provider == "nemotron" else NEMOTRON_MAX_RETRIES

    translated = _translate_nemotron_with_retries(
        text, source_lang, max_retries, base_wait_seconds if provider == "auto" else 5
    )
    if translated:
        return translated

    if provider == "nemotron":
        print(f"Nemotron failed after {max_retries} attempts ({len(text)}-char source). leaving untranslated.")
        return ""

    print(f"Nemotron gave nothing usable for {len(text)}-char source, falling back to fast translator")
    return _translate_fast(text, source_lang)
