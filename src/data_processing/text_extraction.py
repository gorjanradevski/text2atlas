import textract
import subprocess


def pdftotext_v1(path: str) -> str:
    text = textract.process(path, method='tesseract')
    return text


def pdftotext_v2(path: str) -> str:
    import subprocess
    result = subprocess.run(['pdftotext', path, '-'], stdout=subprocess.PIPE)
    return result.stdout
