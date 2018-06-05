"""
This module is created for comparing the results of the manual alignment with those done by computer

compare(man, auto, paragraphs) - ompare the results of one particular manual and one automatic
alignment and return #true_positives, #false_positives, #false_negatives, precision, recall, and fmeasure.

createStatistics(matrix): Return #true_positives, #false_positives, #false_negatives, precision, recall, and fmeasure
for a given matrix.

absp(int par, int sent, boolean inFirstArticle): converts the sentence's position from that given relative to the
beginning of paragraph to that given by the sentence's absolute position in the document

fillMatrix(matrix, lines, startId, changes, paragraphs, stopline="\n"): Read the coordinates from the align algorithm
output file and then change the entries of the matrix that have these coordinates according to the given rules
(the rules are determined by the parameter called "changes")

analyse(slugs, parameters_for_align) - runs align.py on the test data several times until it finds the optimal
constants' values. Takes a lot of time to run.

main(titles, paragraphs, printDetailedStats = True) - get a list of titles and compare pairs of documents with these
titles, some in OUTPUT directory, others in MANUAL directory. If these are files with paragraph alignment,
then paragraphs==True, otherwise paragraphs==False. Return the average fmeasure.
"""

import classpaths as path
import numpy
import re
import align

# during the process of comparison, a matrix is created such that each (i, j) entry maintains information about whether
# the i-th sentence from the first article and the j-th sentence from the second article were aligned manually or by
# computer. The possible values for each entries are:

EMPTY = 0  # the value that is stored in the matrix if the particular pair of sentences was never aligned neither by
# computer, nor manually
AUTO = 1  # if it was only aligned by the computer
MAN = 2  # if it was only aligned manually
MAN_AND_AUTO = 3  # if it was aligned both by the computer and manually
sInd = None  # A tuple of two elements. 0-th element is a list, where for every paragraph in the first article,
# the number of sentences that occurred in a document before the beginning of this paragraph is given. 1-st element
# contains the same information  about the second article. This variable is only used for sentence alignment comparisons


def compare(man, auto, paragraphs):
    """
    Compare the results of one particular manual and one automatic alignment and return precision, recall, and fmeasure
    :param man:     the list of lines of the result obtained by manual alignments
    ;param auto     the list of lines of the results obtained automatically
    :param paragraphs:  true if the files to compare are those with aligned paragraphs. If the files are with aligned
    sentences, then paragraphs == False
    :param printStats:  if true, report calculated results
    :return: precision, recall, fmeasure
    """
    global sInd
    sInd = []
    if paragraphs:
        matrix = numpy.ndarray(list(map(int, re.findall(r'\d+', auto[1]))), numpy.uint8)
        matrix.fill(0)
        fillMatrix(matrix, auto, 2, ((EMPTY, AUTO),), True)
        fillMatrix(matrix, man, 1, ((EMPTY, MAN), (AUTO, MAN_AND_AUTO)), True)
    else:
        sInd.append(list(map(int, re.findall(r'\d+', auto[1]))))
        sInd.append(list(map(int, re.findall(r'\d+', auto[2]))))
        matrix = numpy.ndarray([sInd[0][len(sInd[0]) - 1], sInd[1][len(sInd[1]) - 1]], numpy.uint8)
        matrix.fill(0)
        fillMatrix(matrix, auto, 3, ((EMPTY, AUTO),), False)
        fillMatrix(matrix, man, 1, ((EMPTY, MAN), (AUTO, MAN_AND_AUTO)), False)
    return createStstistics(matrix)


def createStstistics(matrix):
    """
    calculate the precision and recall for a given matrix. Report these if necessary
    :param matrix:      the matrix, which entries are either AUTO (false positives), MAN (false negatives),
                        MAN_AND_AUTO (true positives), or EMPTY
    :param printStats:  if true, print calculated results
    :return: precision, recall, fmeasure
    """
    truePositive = 0
    falsePositive = 0
    falseNegative = 0
    i = 0
    while i < len(matrix):
        j = 0
        while j < len(matrix[i]):
            if matrix[i][j] == AUTO:
                falsePositive += 1
            elif matrix[i][j] == MAN:
                falseNegative += 1
            elif matrix[i][j] == MAN_AND_AUTO:
                truePositive += 1
            j += 1
        i += 1
    if truePositive == 0:
       recall = 0
       precision = 0
    else:
        recall = truePositive / float(truePositive + falseNegative)
        precision = truePositive / float(truePositive + falsePositive)
    if (precision == 0) and (recall == 0):
        fmeasure = 0
    else:
        fmeasure = 2 * (precision * recall) / (precision + recall)

    return truePositive, falsePositive, falseNegative, precision, recall, fmeasure


def absp(values, isFirstArticle):
    """
    Convert the position of the sentence from that represented as a tuple (n_of_paragraph, n_of_the_sentence_in_par)
    to that represented as an integer (number of sentence in the article)
    :param values:          two integers. The first is the paragraph position, the second - the position of sentence
                            relative to the paragraph
    :param isFirstArticle:  whether these coordinates are from the first article or from the second
    :return:                the absolute sentence coordinate
    """
    if isFirstArticle:
        return sInd[0][values[0]-1] + values[1]-1
    else:
        return sInd[1][values[0]-1] + values[1]-1


def fillMatrix(matrix, lines, startId, changes, paragraphs, stopline="\n"):
    """
    Read the coordinates from the align algorithm
    output file and then change the entries of the matrix that have these coordinates according to the given rules
    (the rules are determined by the parameter called "changes"). The new value is determined by the old value. For
    every possible old value there should be a tuple within change parameter, where the first element of the tuple is
    the old parameter, and the second one - the new one
    :param matrix: the matrix to fill
    :param lines:  the lines to read from
    :param startId: the first line to read in lines
    :param changes: the changes as described above
    :param paragraphs: true if the files to compare are those with aligned paragraphs. If the files are with aligned
    sentences, then paragraphs is false
    :param stopline: the line in the input after which the program should stop filling the matrix
    :return: None
    """
    i = startId
    while i < len(lines):
        if lines[i] == stopline:
            break
        if paragraphs:
            line = lines[i].split("\t")
            if len(line) != 3:
                print("every line should contain two sets of numbers. The sets should be separated by the double tab," +
                      " the numbers within the sets - by commas. The line entered was \n"+str(lines[i]))
                break
            first = list(map(int, re.findall(r'\d+', line[0])))
            second = list(map(int, re.findall(r'\d+', line[2])))
            for par0 in first:
                for par1 in second:
                    for change in changes:
                        if matrix[par0-1][par1-1] == change[0]:  # one-indexed to two-indexed
                            matrix[par0-1][par1-1] = change[1]
                            break
        else:
            line = lines[i].split("\t")
            for alignment in line:
                first = absp(list(map(int, re.findall(r'\d+', alignment.split(",")[0]))), True)
                second = absp(list(map(int, re.findall(r'\d+', alignment.split(",")[1]))), False)
                for change in changes:
                    if matrix[first][second] == change[0]:  # one-indexed to two-indexed
                        matrix[first][second] = change[1]
                        break

        i += 1


def main(titles, paragraphs, printDetailedStats = True):
    """
    get a list of titles and compare pairs of documents with these titles, some in the OUTPUT directory, others
    in the MANUAL directory
    :param titles:      the full names of the files (with extension)
    :param paragraphs:  true if the files to compare are those with aligned paragraphs. If the files are with aligned
    sentences, then paragraphs is false
    :param printDetailedStats: if True, print the recall, precision and fmeasure for every single articles. Otherwise,
    only print the average values
    :return: average f measure for all comparisons
    """
    avg_precision = 0  # all the precision values will be added to this value. In the end, this value will be divided by
    # the number of comparisons analysed to report the average precision
    avg_recall = 0
    avg_fmeasure = 0

    if paragraphs:
        manual_directory = path.MANUAL_PARAGRAPHS
        auto_directory = path.OUTDIR_PARAGRAPHS
    else:
        manual_directory = path.MANUAL_SENTENCES
        auto_directory = path.OUTDIR_SENTENCES

    for title in titles:
        with open(manual_directory + title) as m:
            with open(auto_directory + title) as a:
                tPositive, fPositive, fNegative, precision, recall, fmeasure = \
                    compare(m.readlines(), a.readlines(), paragraphs)
                avg_precision += precision
                avg_recall += recall
                avg_fmeasure += fmeasure
                if printDetailedStats:  # then report precision and recall for every individual article
                    print("comparing " + title)
                    print("tp=" + str(tPositive) + " fn=" + str(fNegative) + " fp=" + str(fPositive))
                    print("precision=" + str(round(precision, 5)) + "\t\t recall=" + str(
                        round(recall, 5)) + "\t\t fmeasure=" + str(round(fmeasure, 5)) + " \n\n")
    print("AVERAGE_PRECISION=" + str(round(avg_precision/len(titles), 5)) + "\t\t AVERAGE_RECALL="
              + str(round(avg_recall/len(titles), 5)) + "\t\t AVERAGE_F_MEASURE="
              + str(round(avg_fmeasure/len(titles), 5)))
    return avg_fmeasure/len(titles)


def analize(slugs, parameters_for_align, alpha_variability, alpha2_variability, beta_variability):
    """
    Automatically find the best values for constants in align.py
    :param slugs: the slugs that were aligned manually and automatically (this list is passed as a parameter to the main
    function)
    :param parameters_for_align: essentially the same list but in a different format needed for align.py
    :param alpha_variability: a tuple of three elements. The first element is the minimum value tested, the second value
    is teh maximum, and the third is the step by which the value is incremented from the minimum to the maximum
    :param alpha2_variability: same for alpha2
    :param beta_variability: same for betha
    :return: None
    """
    i = alpha_variability[0]
    bestI = 0
    bestResult = 0
    while i < alpha_variability[1]:
        print("ALPHA="+str(i))
        align.ALPHA = i
        align.align_particular(parameters_for_align)
        current=main(slugs, True, False)
        if current>bestResult:
            bestResult = current
            bestI = i
        i += alpha_variability[2]
    align.ALPHA = bestI
    i = alpha2_variability[0]
    bestI2 = 0
    bestResult2 = 0
    while i < alpha2_variability[1]:
        print("ALPHA2="+str(i))
        align.ALPHA2 = i
        align.align_particular(parameters_for_align)
        current=main(slugs, False, False)
        if current>bestResult2:
            bestResult2 = current
            bestI2 = i
        i += alpha2_variability[2]
    align.ALPHA2 = bestI2
    i = beta_variability[0]
    bestI3 = 0
    bestResult3 = bestResult2
    while i < beta_variability[1]:
        print("BETHA=" + str(i))
        align.BETHA = i
        align.align_particular(parameters_for_align)
        current = main(slugs, False, False)
        if current > bestResult3:
            bestResult3 = current
            bestI3 = i
        i += beta_variability[2]
    print("Best ALPHA="+str(bestI))
    print("Best ALPHA2=" + str(bestI2))
    print("Best BETHA=" + str(bestI3))
    print("Best result=" + str(bestResult3))

if __name__ == "__main__":
    analize(["10dollarbill-woman-cmp-0-1.csv", "ancient-astronomy-cmp-1-2.csv", "ski-swat-cmp-4-5.csv",
             "angrybirds-spying-cmp-2-3.csv", "slavery-reparations-cmp-3-4.csv", "turkey-riots-cmp-0-1.csv",
             "turkey-riots-cmp-1-2.csv", "turkey-riots-cmp-2-3.csv", "turkey-riots-cmp-3-4.csv",
             "pharaoh-tomb-cmp-0-1.csv", "agtech-food-cmp-0-1.csv", "vertical-gardens-cmp-0-1.csv"],
             ["10dollarbill-woman", "ski-swat", "ancient-astronomy", "angrybirds-spying", "slavery-reparations",
              "pharaoh-tomb", "agtech-food", "vertical-gardens", "turkey-riots"], (0.42, 0.58, 0.01),
            (0.30, 0.46, 0.01), (0, 0.4, 0.05))