import os
import time
import requests

NVIDIA_API_KEY = os.environ.get("NVIDIA_API_KEY")
NVIDIA_CHAT_URL = "https://integrate.api.nvidia.com/v1/chat/completions"

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
        "temperature": 1.0,
        "top_p": 0.95,
        "max_tokens": 4096,
        "stream": False,
        "chat_template_kwargs": {"enable_thinking": True},
    }
    response = requests.post(NVIDIA_CHAT_URL, headers=headers, json=payload, timeout=60)
    response.raise_for_status()
    result = response.json()
    message = result["choices"][0]["message"]

    # reasoning_content is a separate field from content on NIM reasoning
    # models; we deliberately never read it for the pasted translation
    reasoning = message.get("reasoning_content")
    if reasoning:
        print(f"  [nemotron reasoning, not used for output]: {len(reasoning)} chars, discarded")

    return message["content"].strip()


def translate_text(text, source_lang="zh", max_retries=5, base_wait_seconds=5):
    attempt = 0
    while attempt < max_retries:
        try:
            translated = translate_text_nemotron(text, source_lang)
            if translated:
                print(f"[nemotron] translated ok ({len(text)} chars source -> {len(translated)} chars output)")
                return translated
        except Exception as e:
            wait_time = base_wait_seconds * (2 ** attempt)
            print(f"Nemotron error ({len(text)}-char source, attempt {attempt + 1}/{max_retries}): {e}. "
                  f"retrying in {wait_time}s")
            time.sleep(wait_time)
            attempt += 1

    print(f"Nemotron failed after {max_retries} attempts ({len(text)}-char source). leaving untranslated.")
    return ""
