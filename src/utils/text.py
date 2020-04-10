import re
import string
from typing import List
from collections import Counter

import numpy as np
import pandas as pd

from utils.general import flatten_list

from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import TweetTokenizer
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS

add_stop = ["", " ", "say", "s", "u", "ap", "afp", "...", "n", "\\"]

stop_words = ENGLISH_STOP_WORDS.union(add_stop)

tokenizer = TweetTokenizer()
pattern = r"(?u)\b\w\w+\b"

lemmatizer = WordNetLemmatizer()

punc = list(set(string.punctuation))


def casual_tokenizer(
    text
):  # Splits words on white spaces (leaves contractions intact) and splits out trailing punctuation
    tokens = tokenizer.tokenize(text)
    return tokens


# Function to replace the nltk pos tags with the corresponding wordnet pos tag to use the wordnet lemmatizer
def get_word_net_pos(treebank_tag):
    if treebank_tag.startswith("J"):
        return wordnet.ADJ
    elif treebank_tag.startswith("V"):
        return wordnet.VERB
    elif treebank_tag.startswith("N"):
        return wordnet.NOUN
    elif treebank_tag.startswith("R"):
        return wordnet.ADV
    else:
        return None


def lemma_wordnet(tagged_text):
    final = []
    for word, tag in tagged_text:
        wordnet_tag = get_word_net_pos(tag)
        if wordnet_tag is None:
            final.append(lemmatizer.lemmatize(word))
        else:
            final.append(lemmatizer.lemmatize(word, pos=wordnet_tag))
    return final


def word_count(text):
    return len(str(text).split(" "))


def word_freq(clean_text_list, top_n):
    """
    Word Frequency
    """
    flat = [item for sublist in clean_text_list for item in sublist]
    with_counts = Counter(flat)
    top = with_counts.most_common(top_n)
    word = [each[0] for each in top]
    num = [each[1] for each in top]
    return pd.DataFrame([word, num]).T


def count_occurrences(text: str, sub_list: List[str]) -> int:
    count: int = 0
    start: int = 0
    sub_list = sorted(sub_list, key=len, reverse=True)
    while True:
        starts = np.array([text.find(sub, start) for sub in sub_list])
        """Find smallest positive one and also with the biggest substring length (ensured by sorting)"""
        if np.all(starts == -1):
            return count
        else:
            index = list(starts).index(min(starts[starts >= 0]))
            start = starts[index] + len(sub_list[index])
            count += 1


def arrange(wordlist, key=len, reverse=True):
    wordlist_new = sorted(wordlist, key=key, reverse=reverse)
    indices = []
    for word in wordlist_new:
        indices.append(wordlist.index(word))
    return wordlist_new, indices


def unarrange(iterable, indices):
    mapping = [(i, indices[i]) for i in range(len(indices))]
    mapping = sorted(mapping, key=lambda x: x[1])
    iterable = [iterable[index] for index, _ in mapping]
    return iterable


def count_occurrences_vec(text: str, sub_list: List[str]) -> List[int]:
    count: np.ndarray = np.array([0] * len(sub_list))
    start: int = 0
    sub_list, indices = arrange(sub_list, key=len, reverse=True)
    while True:
        starts = np.array([text.find(sub, start) for sub in sub_list])
        """Find smallest positive one and also with the biggest substring length (ensured by sorting)"""
        if np.all(starts == -1):
            return unarrange(count, indices)
        else:
            index = list(starts).index(min(starts[starts >= 0]))
            start = starts[index] + len(sub_list[index])
            count[index] += 1


def replace_occurrences(
    text: str, sub_list: List[str], replacement_token: str = " ", sorting: bool = True
) -> str:
    if sorting:
        sub_list = sorted(
            sub_list, key=len, reverse=True
        )  # SORT FROM LONGEST TO SHORTEST - MUST DO THIS, EITHER HERE OR IN PREPROCESSING
    for sub in sub_list:
        text = re.sub(pattern=sub, repl=replacement_token, string=text)
    return text


def delimit_occurrences(text: str, sub_list: List[str], sorting: bool = True) -> str:
    if sorting:
        sub_list = sorted(
            sub_list, key=len, reverse=True
        )  # SORT FROM LONGEST TO SHORTEST - MUST DO THIS, EITHER HERE OR IN PREPROCESSING
    for sub in sub_list:
        text = re.sub(pattern=sub, repl=" " + sub + " ", string=text)
    return text


def detect_occurrences(
    text: str, word_list: List[str], sorting: bool = True
) -> List[str]:
    if sorting:
        word_list = sorted(
            word_list, key=len, reverse=True
        )  # SORT FROM LONGEST TO SHORTEST - MUST DO THIS, EITHER HERE OR IN PREPROCESSING
    hit_list = []
    for word in word_list:
        if word.startswith(" ") or word.endswith(" "):
            matches = re.findall(
                pattern=re.compile("\s*({})\s*".format(word)), string=text
            )
            matches = [match.strip() for match in matches]
        else:
            matches = re.findall(
                pattern=re.compile("\s*(\w*{}\w*)\s*".format(word)), string=text
            )
            matches = [match.strip() for match in matches]
        hit_list.extend(matches)
    hit_list = flatten_list([word.split() for word in hit_list])
    hit_list = [word.replace(",", "").replace(".", "") for word in hit_list]
    hit_list = list(set(hit_list))
    return hit_list


def change_num_token(text: str, new_token="number") -> str:
    text = re.sub("<num>", new_token, text)
    return text
