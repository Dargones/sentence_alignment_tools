"""
This modules contains utils for working with already aligned articles

def get_aligned_sentences(metafile, slug, level1, level2, auto=True): Return aligned sentences..
"""

from newselautil import *
import classpaths as path
import numpy


class Alignment(object):

    """ a class that represents an alignment """

    def __init__(self, sent0, ind0, p_ind0, s_ind0, part0,
                 sent1, ind1, p_ind1, s_ind1, part1):
        """
        All indexes are zero-based.
        :param sent0: the sentence from the first articles
        :param ind0: the absolute index of this sentence in the file
        :param p_ind0: the index of the paragraph this sentence appears in
        :param s_ind0: the index of this sentence relative to the beginning of
        the paragraph. getTokParagraphs(...)[...][p_ind0][s_ind0] will return
        sent0
        :param part0: the part of the sentence that was aligned.
        Parts are separated by semicolons, i.e. sent0.split(';')[part0] is what
        was actually aligned by the algorithm
        :param sent1: same for the second article
        :param ind1:
        :param p_ind1:
        :param s_ind1:
        :param part1:
        """
        self.sent0 = sent0
        self.sent1 = sent1
        self.part0 = part0
        self.part1 = part1
        self.ind0 = ind0
        self.ind1 = ind1
        self.s_ind0 = s_ind0
        self.s_ind1 = s_ind1
        self.p_ind0 = p_ind0
        self.p_ind1 = p_ind1


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


def get_aligned_sentences(metafile, slug, level1, level2, auto=True):
    """
    Returns the list of Alignment objects.
    :param metafile:        the metafile loaded with newselautils.loadMetafile()
    :param slug:            the slug of the aligned articles
    :param level1:          the lower level of the alignment
    :param level2:          the upper level of the alignment
    :param auto:            true if alignments made by the algorithm are to be
                            loaded, false otherwise (for manual alignemnets)
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
        for j in range(len(article)):
            paragraph = article[j]
            if j == 0:
                paragraph[0] = (paragraph[0], 0, 0)
            else:
                paragraph[0] = (paragraph[0], 0, article[j-1][-1][2] + 1)
            for i in range(len(paragraph)-1):
                paragraph[i+1] = (paragraph[i+1], paragraph[i][1] + len(paragraph[i][0].split(";")),
                                  paragraph[i][2] + 1)

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

                ind0 = allParagraphs[0][first[0]][first[1]][2]
                ind1 = allParagraphs[1][second[0]][second[1]][2]
                sent0 = allParagraphs[0][first[0]][first[1]][0]
                sent1 = allParagraphs[1][second[0]][second[1]][0]
                current.append(Alignment(sent0, ind0, first[0], first[1], first[2],
                                         sent1, ind1, second[0], second[1], second[2]))
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
    # result accounts for N-1, N-N and 1-N alignments. new_result does not
    new_result = []
    for x in result:
        new_result += x
    return new_result


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
    """Example use of get_aligned_sentences"""
    metafile = loadMetafile()
    sentpairs = get_aligned_sentences(metafile, "10dollarbill-woman", 0, 2)
    for alignment in sentpairs:
        if alignment.sent0 != alignment.sent1:
            print("FIRST-SENTENCE:" + str(alignment.ind0) + ':' + str(
                alignment.p_ind0) + ':' + str(alignment.s_ind0) + ':' + str(
                alignment.part0) + ' ' + alignment.sent0)
            print("SECOND-SENTENCE:" + str(alignment.ind1) + ':' + str(
                alignment.p_ind1) + ':' + str(alignment.s_ind1) + ':' + str(
                alignment.part1) + ' ' + alignment.sent1)
            print("\n")