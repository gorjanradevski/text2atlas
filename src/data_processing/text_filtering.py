import re


def process_caption(text: str) -> str:
    text = re.sub(r"[\s\n]", " ", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"[Ff]ig\s*\.*\s*\d+\.*", " ", text)
    text = text.strip()
    return text