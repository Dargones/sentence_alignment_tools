# Utils for newsela articles
# Author: S. Anderson
# Modified: A. Fedchin

import io
import re
import string
import csv
import nltk.data
import regex as re
import classpaths as path
from nltk.tokenize import TreebankWordTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

from nltk.corpus import stopwords
STOPWORDS = stopwords.words('english')
STOPWORDS.append("`s")  # TODO: Should this really be appended?
STOPWORDS.append("n`t")
for i in range(len(STOPWORDS) - 2):
    STOPWORDS.append(STOPWORDS[i][0].capitalize() + STOPWORDS[i][1:])

HDR = ['title', 'filename', 'grade_level', 'language',  'version', 'slug']
Tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
Wordtokenizer = TreebankWordTokenizer()
Lemmatizer = WordNetLemmatizer()
htmltag_rm = re.compile(r'(<!--.*?-->|<[^>]*>)')


def loadMetafile():
    """Return list of dictionaries, one for each English Newsela file."""
    numArticles = 0
    enArticles = 0
    articles = []  # all articles
    with open(path.METAFILE,'r') as meta:
        reader = csv.DictReader(meta, delimiter=',')
        for row in reader:
            numArticles += 1
            if row['language'] == 'en':
                enArticles += 1
                articles.append(row)
    return articles


def cleanSentence(s):
    """Clean one string."""
    # if s[0] == '#': return '' # skip lines that are section titles
    return htmltag_rm.sub('', s)  # remove html tags


def cleanSentences(slist):
    """Strip material from list of sentences."""
    sentlist = []
    for s in slist:
        s1 = cleanSentence(s)
        if s1 == '' or s1.isspace():
            continue  # skip
        sentlist.append(s1)
    return sentlist


# Some articles begin with the name of the city in which the event discussed
# has happened. These "location tags" should not affect alignment and should
# be removed. Here are some abbreviations with which some lines begin but
# which should not be removed.
CAPITALIZED_WORDS = ['A', 'FBI', 'FDA', 'NASA', 'DNA', 'TV', 'UFO']
OTHER_HEADERS = '(SEATTLE - |NEW YORK - |WASHINGTON - |BEIJING - |CHICAGO - |' \
                'GORDONVILLE , Pa. - |LOS ANGELES - |BAGHDAD - |' \
                'SAN JOSE , Calif. - |AMSTERDAM - |BAMAKO , Mali - |' \
                'DAKAR , Senegal - |PARIS - |PARIS , France - ' \
                'WASHINGTON , D.C. - |RIYADH - |SAN FRANCISCO , Calif. - |' \
                'CAIRO - |CHICAGO , Ill. - )'
# these are headers that are different to detect otherwise


def modify_the_header(line):
    """
    delete the not very useful headers such as (SEATTLE, Wash. --)
    :param line: the line to modify
    :return: the modified line
    """
    line = re.sub('### PRO : ', '', line)
    line = re.sub('<.*> ', '', line)
    line = re.sub('### PRO : ', '', line)
    if line.split(' ')[0] not in CAPITALIZED_WORDS:
        line = re.sub('^[A-Z][A-Z\-.]* .*?-- ', '', line)
        line = re.sub('^' + OTHER_HEADERS, '', line)
    # firstWord == "!\n":  # a header with an image
    # TODO
    return line


def getTokParagraphs(article, separateBySemicolon=True, MODIFY_HEADER=True):
    """
    Return list of paragraphs.  Each par is a list of strings, each of
    which is an already tokenized sentence.  File suffix should be .tok
    :param article:
    :param separateBySemicolon: if True, the parts of one sentence separated by
    a semicolon will be considered as separate sentences
    :param MODIFY_HEADER: whether the program should 'clean' the headers
    :return:
    """
    SUFFIX = ".tok"
    PARPREFIX = "@PGPH "  # Delimits paragraphs in FILE.tok
    pars = []
    slist = []
    with io.open(path.BASEDIR + '/articles/' + article['filename']+SUFFIX,
                 mode='r', encoding='utf-8') as fd:
        lines = fd.readlines()
        if MODIFY_HEADER:
            lines[1] = modify_the_header(lines[1])
        for i in range(len(lines)):
            if separateBySemicolon:
                phrases = lines[i].split(";")
                for phrase in phrases:
                    if phrase[0:len(PARPREFIX)] == PARPREFIX:  # new paragraph
                        cleaned=cleanSentences(slist)
                        if len(cleaned) > 0:
                            pars.append(cleaned)
                            slist = []
                    else:
                        slist.append( phrase.rstrip('\n'))
            else:
                if lines[i][0:len(PARPREFIX)] == PARPREFIX:
                    # without considering ";" to be a delimiter
                    cleaned = cleanSentences(slist)
                    if len(cleaned) > 0:
                        pars.append(cleaned)
                        slist = []
                else:
                    slist.append(lines[i].rstrip('\n'))
        cleaned = cleanSentences(slist)
        if len(cleaned) > 0:
            pars.append(cleaned)
    return pars


PENNPOS = ['N', 'V', 'J', 'R']
WNETPOS = [wordnet.NOUN, wordnet.VERB, wordnet.ADJ, wordnet.ADV]


def convertPOS(pos):
    """Convert pos to wordnet pos"""
    for i in range(len(PENNPOS)):
        if pos[0] in PENNPOS[i]:
            return WNETPOS[i]
    return None


def lemmatize(s):
    """Return list of lemmas for string s, a sentence."""
    tokens = Wordtokenizer.tokenize(s)
    # tokens = [x for x in tokens if x.lower() == x]
    # remove any string with uppercase char
    # (eg, proper names)
    cleantokens = []
    for w in tokens:
        try:
            w_asc = w.encode('ascii')
            if w not in string.punctuation:
                cleantokens.append(w)
        except UnicodeEncodeError:
            # print "Not ascii:", repr(w)
            pass
    w_tagged = nltk.pos_tag(cleantokens)
    lemmas = []
    for word, pos in w_tagged:
        wordnetPOS = convertPOS(pos)
        if wordnetPOS is None:
            lemmas.append(Lemmatizer.lemmatize(word))
        else:
            lemmas.append(Lemmatizer.lemmatize(word, pos=wordnetPOS))
    i = 0
    # while i<len(lemmas):
    # lemmas[i]=lemmas[i].lower()
    # i += 1
    return lemmas
