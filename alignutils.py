"""
This modules contains utils for working with already aligned articles

def get_aligned_sentences(metafile, slug, level1, level2, auto=True): Return aligned sentences..
"""

from newselautil import *
import classpaths as path
import numpy

class Alignment(object):
    """ a class that represents an alignment """
    def __init__(self, s0, s1, s0part, s1part):
        """
        :param s0: the sentence from the first articles
        :param s1: the sentence from the second article
        :param s0part: the part of the first sentence aligned (parts are separated by semicolons)
        :param s1part: the part of the second sentence aligned
        """
        self.s0 = s0
        self.s1 = s1
        self.s0part = s0part
        self.s1part = s1part


def get_lowest_element_with_slug(slug, metafile):
    """
    return the position of the first element with the given slug within the metafile. Performs the binary search
    :param slug: the slug to search for
    :param metafile: the metafile to use
    :return: The position of the first element with this slug
    """
    hi = len(metafile) - 1  # search for slug
    lo = 0
    while lo < hi:
        mid = int((hi + lo) / 2)
        if metafile[mid]['slug'] < slug:
            lo = mid + 1
        else:
            hi = mid
    if metafile[lo]['slug'] != slug:
        print("No such slug: " + slug)
        return
    # ASSERT: lo contains the slug
    while (lo > 0) and (
        metafile[lo]['slug'] == metafile[lo - 1]['slug']):  # lo should point to the first article with slug
        lo -= 1
    return lo


def replace(threeDimArray, old, new):
    """
    Get a two dimensional array and replace all the old values with the new ones
    :param threeDimArray: the array to interate over
    :param old: the old value
    :param new: the new value
    :return: None
    """
    for twoDim in threeDimArray:
        for oneDim in twoDim:
            for i in range(len(oneDim)):
                if oneDim[i] == old:
                    oneDim[i] == new


def get_aligned_sentences(metafile, slug, level1, level2, paragraphWise = False, auto=True):
    """
    Returns the list of blocks of Alignments. A block of alignment is a set of aignments that share sentences, i.e.
    all of the 1-N or N-N alignments are in the same block and every 1-1 alignment is a block with one element. Hence,
    the returned structure is a list of lists of Alignments.
    :param metafile:        the metafile loaded with newselautils.loadMetafile
    :param slug:            the slug of the aligned articles
    :param level1:          the lower level of the alignment
    :param level2:          the upper level of the alignment
    :param auto:            true if alignments made by algorithm are to be loaded, false otherwise
    :return:
    """
    if level1 >= level2:
        print("level1 is greater than level2")
        return
    lo = get_lowest_element_with_slug(slug, metafile)
    if lo is None:
        return
    allParagraphs = [getTokParagraphs(metafile[lo + level1], False, False),
                     getTokParagraphs(metafile[lo + level2], False, False)]
    result = []

    for article in allParagraphs:
        for paragraph in article:
            paragraph[0] = (paragraph[0], 0)
            for i in range(len(paragraph)-1):
                paragraph[i+1] = (paragraph[i+1], paragraph[i][1] + len(paragraph[i][0].split(";")))

    sentCount = ([], [])
    for i in range(len(allParagraphs)):
        for j in range(len(allParagraphs[i])):
            sentCount[i].append(numpy.ndarray(len(allParagraphs[i][j]),numpy.int8))
            sentCount[i][j].fill(-1)
    # sentCount[0][i][j] is the block of alignment in which the i-th sentence of the j-th paragraoh of the first article
    #  appears. sentCount[0][i][j] is the same thing for the second article
    # if the same sentence appears in two blocks of alignment, the blocks are concatenated

    if auto:
        directory = path.OUTDIR_SENTENCES
        i = 3
    else:
        directory = path.MANUAL_SENTENCES
        i = 1

    with open(directory + slug+"-cmp-"+str(level1)+"-"+str(level2)+".csv") as file:
        f = file.readlines()
        while i < len(f):
            line = f[i].split("\t")
            current = []
            blockId = len(result)  # current is added to result[oldBlock]
            for alignment in line:
                alignment = alignment.split(",")
                first = convert_coordinates(list(map(int, re.findall(r'\d+', alignment[0]))), allParagraphs[0])
                second = convert_coordinates(list(map(int, re.findall(r'\d+', alignment[1]))), allParagraphs[1])

                if blockId == len(result):
                    if (sentCount[0][first[0]][first[1]] != -1)and(sentCount[0][first[0]][first[1]] != len(result)):
                        blockId = sentCount[0][first[0]][first[1]]
                        replace(sentCount, len(result), blockId)
                    elif (sentCount[1][second[0]][second[1]] != -1)and(sentCount[1][second[0]][second[1]]!=len(result)):
                        blockId = sentCount[1][second[0]][second[1]]
                        replace(sentCount, len(result), blockId)
                if sentCount[0][first[0]][first[1]] == -1:
                    sentCount[0][first[0]][first[1]] = blockId
                elif sentCount[0][first[0]][first[1]] != blockId:
                    current += result[sentCount[0][first[0]][first[1]]]
                    result[sentCount[0][first[0]][first[1]]] = None
                    replace(sentCount, sentCount[0][first[0]][first[1]], blockId)
                if sentCount[1][second[0]][second[1]] == -1:
                    sentCount[1][second[0]][second[1]] = blockId
                elif sentCount[1][second[0]][second[1]] != blockId:
                    current += result[sentCount[1][second[0]][second[1]]]
                    result[sentCount[1][second[0]][second[1]]] = None
                    replace(sentCount, sentCount[1][second[0]][second[1]], blockId)

                current.append(Alignment(allParagraphs[0][first[0]][first[1]][0],
                                allParagraphs[1][second[0]][second[1]][0], first[2], second[2]))
            if blockId == len(result):
                result.append(current)
            else:
                result[blockId] += current
            i += 1
        i = 0
        while i < len(result):
            if result[i] is None:
                del result[i]
            else:
                i += 1
    return result

def convert_coordinates(old,pars):
    """
    Convert coordinates from those written in the -cmp- files to those needed in alignutils. First of all, the
    coordinates are made zer0-based instead of 1-based. Secondly, the part of the sentence separated by a semicolon
    are no longer treated as separated sentences
    :param old: old coordinates (n_of_paragraph, n_of_phrase)
    :param pars: the paragraphs for the article for which the coordinates are needed
    :return: new coordinates (n_of_paragraph, n_of_sentence, n_of_phrase)
    """
    old = (old[0]-1, old[1]-1)
    i = 0
    while (i < len(pars[old[0]])-1)and(old[1] >= pars[old[0]][i+1][1]):
        i += 1
    return old[0], i, old[1]-pars[old[0]][i][1]


if __name__ == "__main__":
    # debugging
    metafile = loadMetafile()
    sentpairs = get_aligned_sentences(metafile, "cavemen-recycling", 0, 3)
    for block in sentpairs:
        for alignment in block:
            if alignment.s0 != alignment.s1:
                print("FIRST-SENTENCE: " + alignment.s0)
                print("SECOND-SENTENCE: " + alignment.s1)
        print("\n")
    """i = 0
    nSlugs = 0
    while (nSlugs < 300):
        artLow = i  # first article with this slug
        slug = metafile[i]['slug']
        nSlugs += 1
        while i < len(metafile) and slug == metafile[i]['slug']:
            i += 1
        artHi = i-1
        print ("processing slug: "+ slug)
        while artLow<artHi:
            print(str(artHi-artLow-1)+" "+str(artHi-artLow))
            sentpairs = get_aligned_sentences(metafile, slug, artHi-artLow-1, artHi-artLow)
            artLow +=1"""
