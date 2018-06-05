# Tokenize and Parse a bunch of newsela articles and write to stdout.

import sys
import newselautil as nsla
import classpaths as path
import StanfordParse

def all():
    """
    tokenize all articles
    :return: None
    """
    articles = nsla.loadMetafile()
    i = 0
    # process articles by slug
    while i < len(articles):
        slug = articles[i]['slug']
        NOfLevels = 1
        while (i < len(articles))and(articles[i+1]['slug'] == slug):
            NOfLevels += 1
            i += 1
        processFile(slug, NOfLevels)
        print ('Parsing:' + slug+' '+str(round(i/float(len(articles)), 5))+' of the task completed')
        i += 1

def particular(needed):
    """
    tokenized files with specified slugs
    :param needed: the list of slugs to tokenize
    :return: None
    """
    for slug in needed:
        processFile(slug)


def processFile(slug,numberOfLevels = 6):
    """
    tokenize the files with a given slug
    :param slug: the slug to tokenize
    :param numberOfLevels: specified number of Levels to process
    :return: None
    """
    for i in range(numberOfLevels):
        try:
            StanfordParse.tokenize(path.BASEDIR + '/articles/' + slug + ".en." + str(i) + ".txt")
            with open(path.BASEDIR + '/articles/' + slug + ".en." + str(i) + ".txt.tok") as file:
                lines = file.readlines()
                for j in range(len(lines)):
                    lines[j] = lines[j].replace("'", "`")
            with open(path.BASEDIR + '/articles/' + slug + ".en." + str(i) + ".txt.tok", 'w') as file:
                file.writelines(lines)
        except:
            print('ERROR while parsing %s' % (slug))

if __name__ == "__main__":
    processFile("Hurricane-drones")
