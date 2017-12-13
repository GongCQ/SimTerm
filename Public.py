import fileinput
import os

def FileToSet(path):
    s = set()
    for line in fileinput.input(path):
        if line == '':
            continue
        if line[-1] == os.linesep:
            line = line[0 : len(line) - 1]
        s.add(line)
    return s

def ValidWord1(word):
    if not ('\u4e00' <= word[0] <= '\u9fff' or '\u4e00' <= word[-1] <= '\u9fff') and \
       not ('\u0041' <= word[0] <= '\u005a' or '\u0041' <= word[-1] <= '\u005a') and \
       not ('\u0061' <= word[0] <= '\u007a' or '\u0061' <= word[-1] <= '\u007a'): # 头尾都不是汉字或字母
        return False
    return True

def ValidWord(word):
    if len(word) <= 1:
        return False
    return ValidWord1(word)