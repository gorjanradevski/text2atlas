import re
import textract

def pdftotext(path: str) -> str:
    text = textract.process(path)
    return text.decode("utf-8")

# TODO: MUST REFINE BECAUSE CORRESPONDENCE STUFF GETS WEDGED BETWEEN THE ABSTRACT AND THE KEYWORDS AND WHEN ABSTRACT GOES INTO THE SECOND COLUMN IT PICKS UP CORRESPONDENCE AS WELL
def get_abstract(text: str) -> str:
    text = re.split('Key\s*[Ww]ords', re.split(r'\nAbstract', text)[1])[0]
    text = re.sub(r"[\s\n]", " ", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = text.strip()
    return text