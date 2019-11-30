import textract
import subprocess


import os
import re
import string
from typing import List

import nltk
import contractions
import textract
import unidecode
from unidecode import unidecode

from utils.constants import EMAIL_PATTERN, URL_PATTERN, NUMBER_PATTERN
from utils.loadsave import store_json
STOP_WORDS = nltk.corpus.stopwords.words("english")
"""Creating Text Files from PDF"""


def pdftotext_v1(path: str) -> str:
    text = textract.process(path)
    return text


def pdftotext_v2(path: str) -> str:
    import subprocess
    result = subprocess.run(['pdftotext', path, '-'], stdout=subprocess.PIPE)
    return result.stdout


def extract_keywords(text: str) -> List[str]:
    sections = text.split('\n\n')
    try:
        keywords_section = [section for section in sections
                            if section.startswith(('Key words', 'Keywords', 'Key Words'))][0]
        keywords_section = re.sub('Key words:?|Keywords:?|Key Words:?', '', keywords_section)
        keywords = re.split(r"—|-", keywords_section)
        keywords = [keyword.replace('\n', ' ').strip() for keyword in keywords]
        return keywords
    except IndexError:
        return []


def remove_reference_section(text):
    sep = '\nReference'
    try:
        text = text.rsplit(sep, 1)[0]
    except:
        pass
    return text


def remove_disclosures_section(text):
    sep = '\nDisclosure'
    try:
        text = text.rsplit(sep, 1)[0]
    except:
        pass
    return text


def remove_acknowledgements_section(text):
    sep = '\nAcknowledgment'
    try:
        text = text.rsplit(sep, 1)[0]
    except:
        pass
    return text


def remove_funding_section(text):
    sep = '\nFunding'
    try:
        text = text.rsplit(sep, 1)[0]
    except:
        pass
    return text


def remove_keywords_section(text):
    sub = r'\nKeywords[^\n]+\n'
    return re.sub(sub, '', text)


def remove_part_before_abstract(text):
    sep = '\nAbstract'
    try:
        text = text.split(sep, 1)[1]
    except:
        pass
    return text


def remove_section_titles(text):
    section_titles = ['\nMethods', '\nResults', '\nDiscussion', '\nBackground']
    for section_title in section_titles:
        text = re.sub(section_title, '', text)
    return text


def replace_contractions(text: str) -> str:
    """Replace contractions in string of text - e.g. can't -> cannot"""
    text = contractions.fix(text)
    return text


def replace_unicode(text: str) -> str:
    """
    !!!BEST PERFORMED BEFORE REGEX MATCHING!!!
    Convert unicode characters to ascii approximations
    EXAMPLE: Guča (39 matches) -> Guca (191 match)
    NOTE: Gucha (2 matches) - and this should have been the proper one
    """
    text = unidecode(text)
    return text


def replace_quotes(text):
    return re.sub(r'[\"’‘]', '\'', text)


def remove_url(text: str) -> str:
    text = re.sub(URL_PATTERN, " ", text)
    return text


def remove_email(text: str) -> str:
    text = re.sub(EMAIL_PATTERN, " ", text)
    return text


def remove_extra_whitespaces(text: str) -> str:
    text = re.sub(r"[ \t]{2,}", " ", text).strip()
    return text


def remove_punctuation(text: str) -> str:
    table = str.maketrans({key: " " for key in string.punctuation})
    text = text.translate(table)
    return text


def remove_stopwords(text: str) -> str:
    text = " ".join([word for word in text.split() if word not in STOP_WORDS])
    return text


def remove_small_sections(text: str, wordcount_limit: int = 50) -> str:
    paragraphs = text.split('\n')
    return '\n'.join([paragraph for paragraph in paragraphs if len(paragraph.split()) > wordcount_limit])


def remove_unwanted_sections(text: str) -> str:
    text = remove_part_before_abstract(text)
    text = remove_reference_section(text)
    text = remove_disclosures_section(text)
    text = remove_funding_section(text)
    text = remove_acknowledgements_section(text)
    text = remove_keywords_section(text)
    text = remove_section_titles(text)
    return text


def merge_paragraphs(text):
    return re.sub(r'(\S)\n(\S)', r'\1 \2', text)


def fix_hyphenated_words(text):
    return text.replace('-\n', '')


def remove_citations(text: str) -> str:
    text = re.sub(r"\[(\s*<num>\s*,*\s*)+\]", " ", text)
    return text


def replace_numbers_with_token(text: str, token: str = " number ") -> str:
    text = re.sub(NUMBER_PATTERN, token, text)
    return text


def process_pdf(path: str, output_path: str, converter_version: str = "v1") -> None:
    assert converter_version in ["v1", "v2"], "Invalid argument for converter_version, expected one of {\"v1\", \"v2\"}"
    if converter_version == "v1":
        text = pdftotext_v1(path).decode("utf-8")  # SLOW, ACCURATE
    else:
        text = pdftotext_v2(path).decode("utf-8")  # FAST, MISSES KEYWORDS

    text = replace_unicode(text)
    text = fix_hyphenated_words(text)

    keywords = extract_keywords(text)
    keywords = [keyword for keyword in keywords if keyword]

    text = remove_unwanted_sections(text)
    text = remove_email(text)
    text = remove_url(text)
    text = replace_contractions(text)
    text = merge_paragraphs(text)
    text = remove_small_sections(text)
    text = remove_citations(text)
    text = replace_numbers_with_token(text, token=" number ")
    text = remove_extra_whitespaces(text)
    text = replace_quotes(text)

    data = {"pdf": path, "keywords": keywords, "abstract": '', "fulltext": text}
    store_json(data, output_path)



