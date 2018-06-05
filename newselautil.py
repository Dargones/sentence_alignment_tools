# Utils for newsela articles
# The main collects article stats to be combined with parse information.
# You must run the tokenizer on all files you want to
# process prior to using main.
# Au: S. Anderson
# Modified: A. Fedchin

import io
import re
import string
import csv
import nltk.data
import regex as re
from nltk.tokenize import TreebankWordTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

from nltk.corpus import stopwords
STOPWORDS = stopwords.words('english')
STOPWORDS.append("`s")
STOPWORDS.append("n`t")
length = len(STOPWORDS)-2
for i in range(length):
    wordBeginningWithCapital = STOPWORDS[i][0].capitalize() + STOPWORDS[i][1:]
    STOPWORDS.append(wordBeginningWithCapital)
import classpaths as path

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
    # print 'English %d Total %d' % (enArticles,numArticles)
    return articles


def cleanSentence(s):
    """Clean one string."""
    # if s[0] == '#': return '' # skip lines that are section titles
    return htmltag_rm.sub('', s)  # remove html tags


def cleanSentences(slist):
    """Strip material from list of sentences."""
    DASH = u"\u2014"
    sentlist = []
    for s in slist:
        s1 = cleanSentence(s)
        if s1 == '' or s1.isspace():
            continue  # skip
        sentlist.append(s1)
    return sentlist


# Some articles begin with location name that is written in all caps. The
# location has something to do with the topic of the article, but it is appended
# to any sentence that happens to be the first one. It should not be taken into
# account during the alignment. However, some articles begin with other
# capitalized words that should be taken into account. Here is the list of these
CAPITALIZED_WORDS = ['A', '``', 'U.S.', 'C.J.', 'U.K.', 'FBI', 'FDA', '###', '_',
                    'NASA', 'DNA', 'TV', 'UFO', 'B.B.', '!\n']
LINES_TO_IGNORE = ["## Preamble\n", "CUPERTINO , Calif. .\n",
                   "CHARLESTON , W.Va .\n", "PORTLAND , Ore. .\n",
                   "KANSAS CITY , Missouri .\n", "WICHITA , Kan. .\n",
                   "CHICAGO _ As politicians in Washington took a step toward tightening the nation `s gun laws on Wednesday , first lady Michelle Obama sat down with Chicago high school students whose stories about violence brought her to tears .\n"]


def modify_the_header(line):
    """
    delete the not very useful headers such as (SEATTLE, Wash. --)
    :param line: the line to modify
    :return: the modified line
    """
    # print(line + "\n")
    # isUpperCase = True
    if line not in LINES_TO_IGNORE:
        re.sub('### PRO : ', '', line)
        re.sub('<.*> ', '', line)
        re.sub('### PRO : ', '', line)
        if line.split(' ')[0] not in CAPITALIZED_WORDS:
            re.sub('^[A-Z][A-Z]* ([^-].?|.?[^-])*[^-]*-- ', '', line)
    """firstWord = line.split(' ')[0]
    for letter in firstWord:
        if letter != letter.upper():
            isUpperCase = False
    if isUpperCase and firstWord not in CAPITALIZED_WORDS:
        i = 0
        while ((i < len(line)) and (line[i] != "-")) or ((i + 1 < len(line)) and (line[i+1] == "-")):
            i += 1
        print(line[:i + 2])
        line = line[i+2:]
    elif firstWord == "###":  # ### PRO :
        print(line[:10])
        line = line[10:]
    elif firstWord == '_':
        print(line[:2])
        line = line[2:]
    elif firstWord == "!\n":  # a header with an image
        # TODO
        pass
    if line[0] == '<':  # another version of a header with an image
        i = 0
        while (i < len(line)) and (line[i] != '>'):
            i += 1
        if i+2 < len(line):
            print(line[:i + 2])
            line = line[i + 2:]
            return modify_the_header(line)"""
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
    with io.open(path.BASEDIR + '/articles/' + article['filename']+SUFFIX, mode='r', encoding='utf-8') as fd:
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


def removeCaps(lst):
    return [x for x in lst if x.lower() == x]


def lemmatize(s):
    """Return list of lemmas for string s, a sentence."""
    tokens = Wordtokenizer.tokenize(s)
    # tokens = removeCaps(tokens) # remove any string with uppercase char
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
