from typing import Optional
import jieba
import re
import unicodedata

def clean_text(text: str):
    if not isinstance(text, str):
        text = str(text)
    text = text.replace('\n', " ")  # 新行，我们是不需要的
    text = re.sub("\d+", '', text)  # 删除数字
    text = re.sub('[a-zA-Z]', '', text)  # 删除字母
    text = re.sub('[\s]', '', text)  # 删除空格
    print(text)
    pure_text = ''
    # 以防还有其他特殊字符（数字）等等，我们直接把他们loop一遍，过滤掉
    for letter in text:
        if letter.isalpha() or letter == ' ':
            pure_text += letter
    text = ' '.join(word for word in pure_text.split() if len(word) > 1)
    return text


def cut_text(text: str) -> str:
    text_with_spaces = ''
    text_cut = jieba.cut(text)
    for word in text_cut:
        text_with_spaces += (word) + ' '
    return text_with_spaces



