import re
from typing import List

import numpy as np


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


def detect_occurrences(text: str, word_list: List[str]) -> List[str]:
    word_list = sorted(word_list, key=len, reverse=True)
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
    hit_list = [word.split() for word in hit_list]
    hit_list = [item for sublist in hit_list for item in sublist]
    hit_list = [word.replace(",", "").replace(".", "") for word in hit_list]
    hit_list = list(set(hit_list))
    return hit_list
