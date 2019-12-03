import re
import textract


EMAIL_PATTERN = r"[a-zA-Z0-9+_\-\.]+@[0-9a-zA-Z][.-0-9a-zA-Z]*.[a-zA-Z]+"


def pdftotext(pdf_path: str) -> str:
    text = textract.process(pdf_path)
    return text.decode("utf-8")


def merge_paragraphs(text):
    return re.sub(r"(\S)\n(\S)", r"\1 \2", text)


def remove_correspondence_sections(text: str) -> str:
    paragraphs = text.split("\n")
    return "\n".join(
        [
            paragraph
            for paragraph in paragraphs
            if not re.findall(EMAIL_PATTERN, paragraph)
        ]
    )


def remove_small_sections(text: str, wordcount_limit: int = 50) -> str:
    paragraphs = text.split("\n")
    return "\n".join(
        [
            paragraph
            for paragraph in paragraphs
            if len(paragraph.split()) > wordcount_limit
        ]
    )


def get_pdf_abstract(pdf_path: str) -> str:
    text = pdftotext(pdf_path)
    text = re.split("Key\s*[Ww]ords", re.split(r"\nAbstract", text)[1])[0]
    text = merge_paragraphs(text)
    text = remove_correspondence_sections(text)
    text = remove_small_sections(text)
    text = re.sub(r"[\s\n]", " ", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = text.strip()
    return text
