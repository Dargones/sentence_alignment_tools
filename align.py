"""
Implements the algorithm described by Gustavo Herique Paetzold and Lucia Specia
in "Vicinity-Driven Paragraph and Sentence Alignment for Comparable Corpora", University of Sheffield, UK.


absp(int par, int sent, boolean inFirstArticle): converts the sentence's position given relative to the beginning of
paragraph to that given by the sentence's absolute position in the document


relp(int sent, boolean inFirstArticle): inverse of absp


calculate_cosine_similarity(v0, v1): calculate the cosine similarity between two given vectors


rel_sent_sim(int p0, int s0, int p1, int s1, v0, v1): calculate the cosine similarity between two given sentences,
if it was not calculated previously. Return the calculated value. Use relative sentence coordinates.


abs_sent_sim(sent0, sent1, v0, v1, ind0, ind1): Calculate the cosine similarity between two given sentences, if it was
not calculated previously. Return the calculated value. Use absolute sentence coordinates.

compare_sent_sim(sent0, sent1, v0, v1, ind00, ind01, ind10, ind11): Returns the difference between the cosine
similarities of the two pairs of sentences. Uses absolute sentence coordinates


build_tf_idf(rawVectors, parFq, totalW): get the TF vectors for some set of sentences and a TF vector for the paragraph(s)
that contains all these sentences. Return the TF-IDF vectors


paragraph_similarity(ind0, ind1): calculate the similarity between two given paragraphs if it was not calculated
previously. Return the calculated value.


create_sentence_alignment(tuple_of_coordinates start, next, lists aligned0, aligned1, lists_of_lists_of_vectors v0, v1,
arrays sent0, sent1): This method is called every time a new sentence alignment is found.


align_sentences(sent0, sent1, v0, v1): Takes an array with sentence indexes and TF_IDF vectors as input and uses
(paper: Algorithm2:Sentence Alignment) algorithm to find the alignments among them.


sentence_function(last, next, pars): Function used in euclidean.closest for sentence alignment.


pars_to_sents(pars, sentInd): receive the list of paragraphs' indexes and convert it to an array of absolute
indexes of sentences that appear in these paragraphs


clean_sent_matrix(list_of_ints pars0, list_of_ints pars1): delete all previously calculated entries in the sentSim
matrix if they represent similarities between the sentences from given paragraphs


add_freq(indexes, originals): add together the entries at the same positions in the parFreq arrays. Conceptually,
it merges the term frequency statistics of a set of paragraphs so that the whole set can be perceived as one paragraph.


merge_lists(indexes, originals): merges lists that occupy specified positions in variable originals, which
should itself be a list. Returns one list


create_paragraph_alignment(last, next, pars0, pars1): Create a paragraph alignment. Update and return new last variable.
If this is one-to-one alignment - call align_sentences method.


paragraph_function(last, next):Function used in euclidean.closest for paragraph alignment


align_paragraphs(int a0, int a1): aligns paragraphs using the Vicinity-Driven Paragraph Alignment algorithm.


delete_stopwords(words): Take a list of words and return the list of all words in this list which are not in stopwords


fill_dictionary(dict, parFreq, wordsTotal, article, sentVectors, wordCount): Fill the ditionary with the words from the
given article. Return the number of distinct words in the dictionary. Calculate the term frequency for the paragraphs,
the number of words in each paragraph, and also the term frequency for the sentences


set_up(a0, a1): get the text of the two articles and set up all the arrays and
lists that will be needed later during the alignment.


write_result(slug, loLevel, hiLevel, allParagraphs): print the results of the alignment
to the files in the output directory


def extract_results(): converts the results of the paragraphs' alignment from the matrix to a list
(from parResultMatrix to parResult)


sim_in_articles(slug, paragraphs, levels):  Pairwise compare the levels (given by the levels parameter) of the
 article, given by paragraphs - the list of the tokenized articles with this slug (obtained
 from newselautils.getTokParagraphs)


align_first_n(nToAlign = -1, levels = [(0, 1, 2), (1, 2, 2), (2, 3, 2), (3, 4, 2), (4, 5, 2)]):  Load the information
about the articles and process n slugs by comparing the levels as specified in the levels variable.


align_particular(slugs, levels = [(0, 1, 2), (1, 2, 2), (2, 3, 2), (3, 4, 2), (4, 5, 2)]):  Does the same as
align_first_n but only for specified slugs
"""

from newselautil import *  # the utils used for processing newsela articles.
import classpaths as path  # info about where various source files are stored on this computer
import euclidean as eu  # the tool for iterating over increasing euclidean distance
import alignutils as autils
import math
import numpy
import copy
import sys
is_py2 = sys.version[0] == '2'
if is_py2:
    import Queue as queue
else:
    import queue as queue

USE_CONCENTRATION = True  # if USE_CONCENTRATION = True, the algorithm will align longer sentences
# with shorter ones, considering the standard deviation of the positions of the words common to both sentences
# within the longer one
CONCENTRATION_MODIFIER = 2  # a constant used in calculate_cosine_similarity, if USE_CONCENTRATION is True.
# Specifically, it should be equal to that value of teh coefficient of variation that is small enough for the alignment
# to be made
ALREADY_ALIGNED = -2  # a negative value distinct from -1 that will be assigned to any pair of sentences (an entry)
# in sentSim if one of the sentences in this pair was already aligned. This is needed to skip these pairs when the
# algorithm is called for the second time on the same articles.
MAXIMUM_PARAGRAPHS = 50  # the maximum number of paragraphs in one article. Needed for creating the array of coordinates
# sorted by euclidean distance. If this value is too small, the program will readjust it, but it is preferable that
# this value is big enough in the beginning.
VICINITIES = [((0, 1), (1, 0), (1, 1)), ((1, 2), (2, 1))]  # vicinities as defined in the paper 
# (3. Paragraph Alignment Algorithm). This list might be freely modified, no other changes in the code are necessary.
SENTENCE_VICINITIES = [((0, 1), (1, 0), (1, 1))]  # same for sentences (4. Sentence Alignment Algorithm)
ALPHA = 0.57  # the minimum similarity between two paragraphs that might result in their alignment. This constant is
# called alpha in the paper (Algorithm 1:Paragraph Alignment chart)
ALPHA2 = 0.38  # same for sentences. This constant is also called alpha in the paper (Algorithm 2:Sentence Alignment
# chart). However, it makes sense to separate these two constants (the fmeaseure is higher this way).
BETHA = 0  # slack value for the 1-N/N-1 sentence alignment. This constant is called betha in the paper
# (Algorithm 2:Sentence Alignment chart).
MAX_WORDS = 10000  # Supposed maximum number of distinct words in the two articles. If this constant is not big enough
# the program will crash
sInd = None  # A tuple of two elements. 0-th element is an array, where for every paragraph in the first article,
# the number of sentences that occurred in a document before the beginning of this paragraph is given. 1-st element -
# the same for the second article.
sCoor = None  # For every article and for every sentence in the article, the index of the paragraph this sentence
# appears in is stored.
parSim = None  # the similarity matrix for the paragraphs. It is referred as M in the paper (3. Paragraph Alignment)
sentSim = None  # the similarity matrix for sentences. It is also referred as M in the paper (3. Sentence Alignment)
# In the article, the authors suggest to create a new sentSim matrix every time the sentence alignment method is called.
# However, most of the time the paragraphs are aligned one-to-one. Hence, the cosine similarities between sentences that
# were calculated during the paragraph alignment can usually be reused during the sentence alignment. Therefore, sentSim
# matrix stores the information about all the alignments made previously
v = None  # For every article, for every paragraph in the article and for every sentence in the paragraph, there is a
# vector stored in v variable. The vector lies in R_n, where n is the number of words in the text. In order to
# reduce the number of operations in built_tf_idf and calculate_cosine_similarity, only non-zero entries are stored.
# Hence, a "vector" consists of multiple tuples. The first value in a tuple stores the index related to the
# word, the second one - the position of the word within the sentence. If the word occurs more than once within the same
# sentence, it occupies more than one entry in the vector so that the positions could be stored. To get the term
# frequency for a certain word, the number of these entries should be calculated. The entries are sorted by the indexes
# related to distinct words.
wordsTotal = None  # the total number of words in every paragraph. wordsTotal[0] stores the info about the first article
# , wordsTotal[1] - about the second
parFreq = None  # for every paragraph in each article the number of times each word appears in the paragraph is stored.
parResultMatrix = None  # the matrix that represents paragraph alignments. If the two paragraphs are aligned, the entry
# will be True, otherwise - False
parResult = None  # the list of paragraph alignments made. Every i-th element of the list is a tuple of two elements.
# The 0th element is the list of indexes of the paragraphs from the first article that are part of the i-th alignment. 
# The 1-st element is the list of indexes of the paragraphs from the second article that are part of the i-th alignment.
result = None  # the same for sentences. A sentence index is given as a tuple (par_index,sentence_in_par_index).


def absp(par, sent, inFirstArticle):
    """
    Convert the position of the sentence from that represented as a tuple (n_of_paragraph, n_of_the_sentence_in_par)
    to that represented as an integer (number of sentence in the article)
    :param par:             the position of the paragraph the sentence appears in (int)
    :param sent:            the position of the sentence relative to the beginning of the paragraph (int)
    :param inFirstArticle:  whether this sentence is in the first article or in the second (boolean)
    :return: an integer that represents the absolute position of a sentence, i.e. the position of sentence within
     the article
    """
    if inFirstArticle:
        return sInd[0][par] + sent
    else:
        return sInd[1][par] + sent


def relp(sent, infirstArticle):
    """
    Convert the position of the sentence from that given by the sentence's index in the document to that given by
    a tuple (n_of_paragraph, n_of_the_sentence_in_par).
    :param sent:            sentence's index in the document (int)
    :param infirstArticle:  whether this sentence is in the first article or in the second (boolean)
    :return: relative coordinate of the sentence, i.e. the tuple (n_of_paragraph, n_of_the_sentence_in_par)
    """
    if infirstArticle:
        return sCoor[0][sent], sent - sInd[0][sCoor[0][sent]]
    else:
        return sCoor[1][sent], sent - sInd[1][sCoor[1][sent]]


def calculate_cosine_similarity(v0, v1):
    """
    Calculate the cosine similarity between the two given sentences. The sentences should be given in the form of
    vectors. A vector consists of multiple tuples. The first value in a tuple stores the index related to the word,
    the second one - precalculated TF-IDF, the third one - the position of the word within the sentence. If the word
    occurs more that once within the same sentence, two or more entries will be needed to store the information about
    this word. In this case, TF_IDF can be extracted by summarizing all the TF_IDFs in these entries (which will be
    equal to each other). The number of entries in the first vector should not be smaller than that in the second one.
    Otherwise, the function will call itself reversing the order of the parameters
    :param v0: the vector for the first sentence
    :param v1: same for the second
    """
    if (len(v0) == 0)or(len(v1) == 0):
        return 0
    elif len(v0) < len(v1):
        return calculate_cosine_similarity(v1,v0)
    i = 0  # the index to iterate over entries in the first vector
    j = 0  # same for the second vector
    dotProduct = 0  # the dotProduct of two vectors
    lenS = 0  # the length of the smaller vector squared
    lenLCommon = 0
    lenLDistinct = 0  # lenLCommon + lenLDistinct is the length of the longer vector squared. lenLCommon is the part
    # that is due to the words common to both vectors, lenLDistinct - the other half
    positions = []  # the list of positions of the words in the longer vector which also occur in the second one
    sum = 0  # the sum of these positions
    while (i < len(v0)) and (j < len(v1)):
        if v1[j][0] <= v0[i][0]:  # v1[j][0] wither does not appear in v0, or this word appears in both sentences
            val1 = v1[j][1]  # val1 will store the TF-IDF for a given word
            while (j + 1 < len(v1)) and (v1[j][0] == v1[j + 1][0]):  # summarizing TF-IDF values of all words in the
                # shorter vector
                j += 1
                val1 += v1[j][1]
            lenS += val1 * val1
            if v0[i][0] == v1[j][0]:  # IF THIS WORD IS COMMON TO BOTH SENTENCES
                val0 = v0[i][1]
                positions.append(v0[i][2])
                sum += v0[i][2]
                while (i+1 < len(v0)) and (v0[i][0] == v0[i+1][0]):  # summarizing TF-IDF values of all identical words
                    # in the longer vector
                    i += 1
                    val0 += v0[i][1]
                    positions.append(v0[i][2])
                    sum += v0[i][2]
                lenLCommon += val0 * val0
                dotProduct += val0 * val1
                i += 1
            j += 1
        else:
            val0 = v0[i][1]
            while (i+1 < len(v0)) and (v0[i][0] == v0[i+1][0]):  # summarizing TF-IDF values of all distinct words
                # in the longer vector
                i += 1
                val0 += v0[i][1]
            lenLDistinct += val0 * val0
            i += 1
    while i < len(v0):
        val0 = v0[i][1]
        while (i + 1 < len(v0)) and (v0[i][0] == v0[i + 1][0]):
            i += 1
            val0 += v0[i][1]
        lenLDistinct += val0 * val0
        i += 1
    while j < len(v1):
        val1 = v1[j][1]
        while (j + 1 < len(v1)) and (v1[j][0] == v1[j + 1][0]):
            j += 1
            val1 += v1[j][1]
        lenS += val1 * val1
        j += 1

    if lenLCommon == 0:
        return 0
    if USE_CONCENTRATION:
        if (len(v1) < len(v0) / 2) and (len(positions) > 2):  # in this case it is likely that the shorter vector
            # corresponds to some part of the longer ones, or these vectors don't have anything in common at all
            average = sum / float(len(positions))  # the average of the positions of the words in the longer vector
            # that also occur in the shorter one
            deviation = 0  # this will be the standard deviation of the positions of the words that occur in both
            # sentences, within the longer sentence
            for i in range(len(positions)):
                deviation += math.fabs(positions[i]-average)
            deviation /= float(len(positions))
            c_of_variation = deviation * 2 / len(positions)
            return dotProduct / float(math.sqrt(lenLCommon * lenS)) / c_of_variation / CONCENTRATION_MODIFIER
    return dotProduct / float(math.sqrt((lenLCommon + lenLDistinct) * lenS))


def rel_sent_sim(p0, s0, p1, s1, v0, v1):
    """
    Calculate the cosine similarity between two given sentences, if it was not calculated previously.
    Return the calculated value. Use relative sentence coordinates.
    :param p0: the index of the paragraph the sentence from the first article appears in
    :param s0: the position of this sentence relative to the beginning of the paragraph
    :param v0: the vector representing the sentence from the first article
    :param p1: same as p0, but for the second article
    :param s1: same as s0, but for the second article
    :param v1: same as v0, but for the second article
    :return: the cosine similarity between the two sentences specified
    """
    global sentSim
    if (sentSim[absp(p0, s0, True)][absp(p1, s1, False)] < 0)and(sentSim[absp(p0, s0, True)][absp(p1, s1, False)] !=ALREADY_ALIGNED):
        # if the sentence similarity was not yet calculated.
        sentSim[absp(p0, s0, True)][absp(p1, s1, False)] = calculate_cosine_similarity(v0, v1)
    return sentSim[absp(p0, s0, True)][absp(p1, s1, False)]


def abs_sent_sim(sent0, sent1, v0, v1, ind0, ind1):
    """
    Calculate the cosine similarity between two given sentences, if it was not calculated previously.
    Return the calculated value. Use absolute sentence coordinates.
    :param sent0: the array of absolute coordinates of sentences in the first article
    :param sent1: the array of absolute coordinate of sentences in the secocnd article
    :param v0:    the array of TF-IDF vectors for sentence in the fisrt article
    :param v1:    same for the second article
    :param ind0:  the position of the sentence to compare in sent0 and v0 arrays. Note the ind0 is not a coordinate,
    neither absolute, nor relative
    :param ind1:  same for the second article
    :return:      the cosine similarity between the two sentences specified
    """
    global sentSim
    # ind0 and ind1 are used, to reduce the overload of indexes in align_sentences and create_sentence_alignment methods
    if (sentSim[sent0[ind0]][sent1[ind1]] < 0)and(sentSim[sent0[ind0]][sent1[ind1]] != ALREADY_ALIGNED):  # if the sentence similarity has not yet been calculated.
        sentSim[sent0[ind0]][sent1[ind1]] = calculate_cosine_similarity(v0[ind0], v1[ind1])
    return sentSim[sent0[ind0]][sent1[ind1]]


def compare_sent_sim(sent0, sent1, v0, v1, ind00, ind01, ind10, ind11):
    """
    Return the difference between two sentence similarities
    :param sent0: the array of absolute coordinates of sentences in the first article
    :param sent1: the array of absolute coordinate of sentences in the secocnd article
    :param v0:    the array of TF-IDF vectors for sentence in the first article
    :param v1:    same for the second article
    :param ind00:  the position of the sentence to compare in sent0 and v0 arrays. Note the ind0 is not a coordinate,
    neither absolute, nor relative. This is for the first comparison and the first article
    :param ind01:   same for the second article. This is for the first comparison
    :param ind10:   This is for the first article and the second comparison
    :param ind11:   This is for the second article and the second comparison
    :return:        the difference between the cosine similarities between the two pairs of sentences specified
    """
    return abs_sent_sim(sent0, sent1, v0, v1, ind00, ind01) - abs_sent_sim(sent0, sent1, v0, v1, ind10, ind11)


def build_tf_idf(rawVectors, parFq, totalW):
    """
    Get the TF vectors for some set of sentences and the frequency statistic for words in the paragraph(s) these
    sentences appear in. Return the TF-IDF vectors
    :param rawVectors:  TF vectors. Vector consists of multiple tuples. The first value in a tuple stores the index
    related to the word, the second one - the position of this word within the sentence. If the word appears more than
    once in the same sentence it occupies more than one entry, so that the position of the word within the sentence
    might be stored. Nevertheless, the TF will be calculated correctly in cosine_similarity.
    :param parFq:       for every word in the text stores the number of times this word appears in this paragraph(s)
    :param totalW:      total number of words in this paragraph
    :return:            TF_IDF vectors
    """
    i = 0
    newVector = []
    while i < len(rawVectors):
        newVector.append(numpy.ndarray(len(rawVectors[i]), dtype=[('ind', numpy.uint16), ('freq', numpy.float16),
                                                                  ('pos', numpy.uint16)]))
        j = 0
        while j < len(rawVectors[i]):
            newVector[i][j][0] = rawVectors[i][j][0]  # the word index remains the same
            newVector[i][j][2] = rawVectors[i][j][1]
            newVector[i][j][1] = math.log((totalW+1) / float(parFq[rawVectors[i][j][0]]))
            #  +1 is necessary so that the logarithm will never be zero
            j += 1
        i += 1
    return newVector


def paragraph_similarity(ind0, ind1):
    """
    Calculate the similarity between two given paragraphs (as stated in the paper (3. Paragraph Alignment Algorithm), it
    is the maximum cosine similarity between any pair of sentences from the two paragraphs) if it was not calculated
    previously. Return the calculated value.
    :param ind0: the position of the first paragraph within the first article
    :param ind1: the position of the second paragraph within the second article
    :return: the similarity between them [0,1]
    """
    if parSim[ind0][ind1] < 0:  # if the paragraph similarity was not yet calculated
        TF_IDF_built = False # true if TF_IDF for these paragraphs was already built. If the algorithm is called for
        # the second or the third time, it might be that building TF_IDF will not be needed.
        max = 0  # the maximum cosine similarity found
        i = 0
        while i < len(v[0][ind0]):
            j = 0
            while j < len(v[1][ind1]):
                if sentSim[absp(ind0, i, True)][absp(ind1, j, False)] != ALREADY_ALIGNED:
                    if not TF_IDF_built: # if the similarity between these two sentences was never calculated
                        vectors = (build_tf_idf(v[0][ind0], parFreq[0][ind0], wordsTotal[0][ind0]),
                                   build_tf_idf(v[1][ind1], parFreq[1][ind1], wordsTotal[1][ind1]))  # creating TF-IDF
                        # vectors for this particular set of sentences in these particular paragraphs
                        TF_IDF_built = True
                    if rel_sent_sim(ind0, i, ind1, j, vectors[0][i], vectors[1][j]) > max:
                            max = rel_sent_sim(ind0, i, ind1, j, vectors[0][i], vectors[1][j])
                j += 1
            i += 1
        parSim[ind0][ind1] = max
    return parSim[ind0][ind1]


def create_sentence_alignment(start, next, aligned, v0, v1, sent0, sent1):
    """
    This method is called every time a new sentence alignment is found. If the new alignment is a part of 1-N or N-1
    alignment, the program appends the new sentence to the list of sentences, which constitute this alignment and also
    searches whether further 1-N, N-1 alignments can be made. If the new alignment is 1-1 alignment, the program appends
    this list to the result variable.  (in the paper this is the part of Algorithm 2.Sentence Alignment)
    :param start:       the previous alignment
    :param next:        position of the new alignment relative to the last one
    :param aligned0:    the list of sentences from the first article that were aligned since the last 1-1
    alignment was made
    :param aligned1:    same for the second article
    :param v0:          the vectors that store the precalculated TF_IDF for the sentences in the first article
    :param v1:          same for the second article
    :param sent0:       the array that links the positions indicated via "start" and "next" parameters with the
    corresponding sentences in the sentSim matrix. For example, a sentence that has the position start[0] in v0
    will have ythe position [0][sent0[start[0]]] in sentSim.
    :param sent1:       same for the second article
    :return:            the coordinates of the new alignment
    """
    if (next[0] != 0) and (next[1] != 0):  # if this is true, then this alignment is a start of a new alignment
        global result  # hence the info about the last alignment should be added to the result
        lst = []
        for i in range(len(aligned)):
            lst.append((relp(aligned[i][0], True), relp(aligned[i][1], False)))
        result.append(lst)

        for i in range(len(aligned)):
            for s1 in range(len(sentSim[aligned[i][0]])):
                sentSim[aligned[i][0]][s1] = ALREADY_ALIGNED  # this is needed to speed up the algorithm, when it is
                # called for the second (third) time. The algorithm will ignore all previously aligned sentences
        for s0 in range(len(sentSim)):
            for j in range(len(aligned)):
                sentSim[s0][aligned[j][1]] = ALREADY_ALIGNED

        del aligned[:]
        aligned.append((sent0[start[0] + next[0]], sent1[start[1] + next[1]]))
        return start[0] + next[0], start[1] + next[1]

    elif (next[0] == 0) and (next[1] != 0): # this means that this sentence is a part of 1-N alignment
        n = next[1]+1  # as the authors of the paper suggest, when a 1-N alignment is found, the algorithm tries to
        # "extend" this alignment by considering 1-N+1, 1-N+2 etc. alignments.
        aligned.append((aligned[-1][0], sent1[start[1] + next[1]]))
        while (start[1] + n < len(sent1)) and (compare_sent_sim(sent0, sent1, v0, v1, start[0], start[1] + n, start[0], start[1]+n-1) > - BETHA):
            if (start[0] + 1 < len(sent0)) and (compare_sent_sim(sent0, sent1, v0, v1, start[0], start[1] + n, start[0]+1, start[1] + n) >= 0):
                # in this if statement, the similarity between the sentences that are to be aligned is compared with
                # the similarity between the nearest "diagonal" as it is suggested in the paper
                break;
            aligned.append((aligned[-1][0], sent1[start[1] + n]))
            n += 1
        return start[0], start[1] + n - 1
    elif (next[0] != 0) and (next[1] == 0):  # N-1 alignment
        n = next[0]+1
        aligned.append((sent0[start[0] + next[0]], aligned[-1][1]))
        while (start[0] + n < len(sent0)) and (compare_sent_sim(sent0, sent1, v0, v1, start[0] + n, start[1], start[0] + n - 1, start[1]) > - BETHA):
            if (start[1] + 1 < len(sent1)) and (compare_sent_sim(sent0, sent1, v0, v1, start[0]+n, start[1], start[0]+n, start[1]+1) >=0):
                break;
            aligned.append((sent0[start[0] + n], aligned[-1][1]))
            n += 1
        return start[0] + n - 1, start[1]


def sentence_function(last, next, pars):
    """
    Function used in euclidean.closest for sentence alignment
    :param last: the coordinates of the previous alignemnt made
    :param next: the coordinates of the new cosidered alignment relative to the last one
    :param pars: extra parameters given as a list. Include sent0,sent1,v0 and v1 from the align_sentences method
    :return:     whether these sentences could be aligned (whether the cosine similarity betwee them is greater than
    ALPHA2)
    """
    return abs_sent_sim(pars[0], pars[1], pars[2], pars[3], last[0] + next[0], last[1] + next[1]) > ALPHA2


def align_sentences(sent0, sent1,v0,v1):
    """
    Takes an array with sentence indexes as input and uses (paper: Algorithm2:Sentence Alignment) algorithm to find the
    alignments among them. Firstly, the algorithm looks for the best matching air of sentences within vicinities. If
    the similarity between these two sentences is greater than ALPHA2, the algorithm adds a new alignment. Otherwise,
    it searches for the first pair of sentences that has the similarity of ALPHA2 between them by iterating by
    the euclidian distance from the last alignment. Appends new found alignments to result variable
    :param sent0:   This array is used to convert the indexes of the sentences that are used in the method (which are
    in [0...len(sent0)-1]) to the indexes of the same sentences in sentSim
    :param sent1:   Same for the second article
    :param v0:      vectors that contain precalculated TF_IDF for the sentences from the first article
    :param v1:      same for the seond article
    :return:        None
    """
    eu.calculate(len(sent0), len(sent1), VICINITIES, SENTENCE_VICINITIES) # this line is added to make sure that the
    # array in euclidean is long enough to iterate over sentences in this particular case. The array in euclidean
    # should be long enough and this line should not result in any further calculations
    start = eu.closest((0, 0), 0, len(sent0), len(sent1), sentence_function, [sent0, sent1,v0,v1]) # the first pair of
    # sentences that has the similarity between them great enough
    if start is None:
        return
    aligned = [(sent0[start[0]], sent1[start[1]])] # this will be the current blocks of alignments. A block of
    # alignments is a set of alignments that share sentences. An alignment is a tuple of indexes. Hence, aligned is
    # a list of tuples of indexes
    while True:
        next = SENTENCE_VICINITIES[0][0]
        max = 0
        alignmentMade = False
        for vicinity in SENTENCE_VICINITIES: # searching for similar sentences within vicinities
            for c in vicinity:
                if (start[0] + c[0] < len(sent0)) and (start[1] + c[1] < len(sent1)):
                    if abs_sent_sim(sent0, sent1, v0, v1, c[0] + start[0], c[1] + start[1]) > max:
                        max = abs_sent_sim(sent0, sent1, v0, v1, c[0] + start[0], c[1] + start[1])
                        next = c
            if max > ALPHA2:  # make an alignment
                start = create_sentence_alignment(start, next, aligned, v0, v1, sent0, sent1)
                alignmentMade = True
                break
        if not alignmentMade:  # all vicinities are checked. From this point the algorithm searches for the nearest pair
            # of sentences such that the similarity between them is >ALPHA.
            next = eu.closest(start, eu.sentStart, len(sent0), len(sent1), sentence_function, [sent0, sent1,v0,v1])
            if next is None:
                break
            else:
                start = create_sentence_alignment(start, next, aligned, v0, v1, sent0, sent1)
    create_sentence_alignment(start, (-1,-1), aligned, v0, v1, sent0, sent1) # an extra imaginary alignment
    # is added so that the last real one will be processed. This extra alignment is stored nowhere and is safe to make


def pars_to_sents(pars, sentInd):
    """
    Receive the list of paragraphs' indexes and convert it to an array of absolute indexes of sentences that appear in
    these paragraphs
    :param pars:            the list of paragraphs' indexes
    :param sentInd:         Array that contains the information about the number of sentences that appear before the
                            beginning of each paragraph.
    :return:                the list of sentences' indexes
    """
    len = 0
    for par in pars:
        len += sentInd[par + 1] - sentInd[par]
    sentences = numpy.ndarray(len, numpy.uint16)
    count = 0
    for par in pars:
        i = sentInd[par]
        while i < sentInd[par + 1]:
            sentences[count] = i
            count += 1
            i += 1
    return sentences


def clean_sent_matrix(pars0, pars1):
    """
    Deletes all previously calculated entries in the sentSim matrix if they represent similarities between the sentences
    from given paragraphs
    :param pars0: the list of paragraphs from the first article
    :param pars1: the list of paragraphs from the second article
    :return: None
    """
    for par0 in pars0:
        i = sInd[0][par0]
        while i < sInd[0][par0 + 1]:
            for par1 in pars1:
                j = sInd[1][par1]
                while j < sInd[1][par1 + 1]:
                    if sentSim[i][j] != ALREADY_ALIGNED:
                        sentSim[i][j] = -1
                    j += 1
            i += 1


def add_freq(indexes, originals):
    """
    add together the entries at the same positions in the parFreq arrays. Conceptually, it merges the frequency statistics
    of a set of paragraphs so that the whole set can be perceived as one paragraph.
    :param indexes: indexes of the arrays in originals to add
    :param originals: the actual arrays to add together
    :return: the array of the sums of entries
    """
    destination=copy.deepcopy(originals[indexes[0]])
    j = 1
    while j < len(indexes):
        i = 0
        while i < len(destination):
            destination[i] += originals[indexes[j]][i]
            i += 1
        j += 1
    return destination


def merge_lists(indexes, originals):
    """
    merges lists that occupy specified positions in variable originals, which should itself be a list. Returns one list
    :param indexes: indexes of the lists in originals to merge
    :param originals: the actual lists to add together
    :return: the list that is the sum of the specified lists
    """
    destination = []
    for index in indexes:
        destination += originals[index]
    return destination


def create_paragraph_alignment(last, next, pars0, pars1):
    """
    Create an alignment. Update and return new last variable.
    If this is one-to-one alignment - call align_sentences for the aligned paragraphs and refresh pars0 and pars1.
    If this is 1 to N or N to 1 alignment, add the last[0]+next[0] to pars0 and last[1]+next[1] to pars1.
    :param last:    the last alignemnt made
    :param next:    position of the new alignment relative to the last one
    :param pars0:   list of all paragraphs fro the first aligned since th east 1-to-1 alignment was made
    :param pars1:   same for the second
    :return:        new value for last
    """
    if next[0] != 0:
        if next[1] != 0:  # if True, this is a 1-to-1 alignment
            # global parResult # this works if the algorithm is only called once
            # parResult.append((copy.deepcopy(pars0), copy.deepcopy(pars1)))
            # if (pars0==[2,3,4])&(pars1==[3]):
                # print("debug")
            global parResultMatrix
            for par0 in pars0:
                for par1 in pars1:
                    parResultMatrix[par0][par1] = True
            if len(pars0) + len(pars1) > 2:  # if there were N-1 or 1_N or N-N alignments made since the last call to
                # align_sentences, the sentSim matrix cannot be reused, because the paragraphs should be conceptually
                # concatenated and therefore, IDF changes. The conceptual concatenation of the paragraphs is suggested
                # by the authors of the paper in the end of 3.Paragraph Alignment Algorithm section
                clean_sent_matrix(pars0, pars1)
                parFq = (add_freq(pars0, parFreq[0]), add_freq(pars1, parFreq[1]))
                totalW = [0,0]
                for par in pars0:
                    totalW[0] += wordsTotal[0][par]
                for par in pars1:
                    totalW[1] += wordsTotal[1][par]
                # parFq now contains the term frequency for the "concatenated" paragraphs
                vectors = (build_tf_idf(merge_lists(pars0, v[0]), parFq[0],totalW[0]),
                           build_tf_idf(merge_lists(pars1, v[1]), parFq[1], totalW[1]))  # creating TF-IDF vectors
            else:  # if no concatenation is needed, the program proceeds straight to the creation of TF_IDF vectors
                vectors = (build_tf_idf(v[0][pars0[0]], parFreq[0][pars0[0]], wordsTotal[0][pars0[0]]),
                           build_tf_idf(v[1][pars1[0]], parFreq[1][pars1[0]], wordsTotal[1][pars1[0]]))
            align_sentences(pars_to_sents(pars0, sInd[0]), pars_to_sents(pars1, sInd[1]), vectors[0], vectors[1])
            # TF-IDF vectors are passed as argument to the align_sentences method
            del pars0[:]  # since here pars0 is only a reference, no initialization can be done with it, because it is
            # used in the outer scope
            del pars1[:]  # same for pars1
            pars0.append(last[0] + next[0])
            pars1.append(last[1] + next[1])
        else:  # if it is a 1-N or N-1 alignment, the align_sentences algorithm is not called.
            pars0.append(last[0] + next[0])
    elif next[1] != 0:
        pars1.append(last[1] + next[1])
    return last[0] + next[0], last[1] + next[1]


def paragraph_function(last, next):
    """
    Function used in euclidean.closest for paragraph alignment
    :param last: the coordinates of the previous alignemnt made
    :param next: the coordinates of the new considered alignment relative to the last one
    :return:     whether these paragraphs could be aligned (whether the cosine similarity between them is greater than
    ALPHA)
    """
    return paragraph_similarity(last[0] + next[0], last[1] + next[1]) > ALPHA


def align_paragraphs(a0, a1):
    """
    aligns paragraphs using the Vicinity-Driven Paragraph Alignment algorithm (Algorithm1: Paragraph Alignment Chart)
    :param a0:  the number of paragraphs in the first article
    :param a1:  the number of paragraphs in the second article
    :return:    None
    """
    eu.calculate(a0, a1, VICINITIES, SENTENCE_VICINITIES)  # check if the euclidean distance array is large
    # enough (it should be)
    last = eu.closest((0,0), 0, a0, a1, paragraph_function)  # searching for the first alignment. Unlike the authors of
    # the paper suggest, no assumption is made that the first paragraphs align
    if last is None:
        return
    pars0 = [last[0]]  # all the paragraphs from the first article aligned after last 1 to 1 alignment was made
    pars1 = [last[1]]  # same for the second article
    while True:
        next = VICINITIES[0][0]
        max = 0
        alignmentMade = False
        for vicinity in VICINITIES:  # searching for maximum similarity within all the vicinities. To allow to
            # modify the VICINITIES, the loop is employed instead of the conditional operators. The change in the
            # VICINITIES list will not affect the code
            for c in vicinity:
                if (last[0] + c[0] < a0) and (last[1] + c[1] < a1):
                    if (paragraph_similarity(last[0] + c[0], last[1] + c[1])) > max:
                        max = paragraph_similarity(c[0] + last[0], c[1] + last[1])
                        next = c
            if max > ALPHA:  # if a good enough alignment was found
                last = create_paragraph_alignment(last, next, pars0, pars1)
                alignmentMade = True
                break
        if not alignmentMade:  # all vicinities are checked. From this point the algorithm searches for the nearest pair
            # of paragraphs such that the similarity between them is >ALPHA.
            next = eu.closest(last, eu.parStart, a0, a1, paragraph_function)
            if next is None:
                break
            else:
                last = create_paragraph_alignment(last, next, pars0, pars1)
    create_paragraph_alignment(last, (a0,a1), pars0, pars1)  # an extra imaginary alignment is added so
    # that the last real one will be processed. This extra alignment is stored nowhere and is safe to make


def delete_stopwords(words):
    """
    Take a list of words and return the list of all words in this list which are not in stopwords
    :param words:   the list of words to process
    :return:        the same lists with all stopwords excluded
    """
    result = []
    for word in words:
        if word not in STOPWORDS:
            result.append(word)
    return result


def fill_dictionary(dict, parFreq, wordsTotal, article, sentVectors, wordCount):
    """
    Fill the dictionary with the words from the given article. Return the number of distinct words in the dictionary.
    Calculate the term frequency for the paragraphs, the number of words in each paragraph, and also the term frequency
    for the sentences
    :param dict:        dictionary to fill the words in. Every entry in the dictionary is filled with a value specific
                        to the word. This value will be used elsewhere instead of the string itself.
    :param parFreq:     array that is to be filled with term frequency for paragraphs
    :param wordsTotal:  total number distinct words in each paragraph (is calculated by this method)
    :param article:     the article to process, i.e. its text acquired via newselautils.getTokParagraphs
    :param sentVectors: array that is to be filled with term frequency for sentences
    :param wordCount:   total number of distinct words throughout the text of both articles
    :return:            new value of wordCount
    """
    parN = -1  # number of paragraphs proessed
    for par in article:
        sentVectors.append([])
        parN += 1
        sentN = -1  # number of sentences processed in this paragraph
        for sent in par:
            words = delete_stopwords(lemmatize(sent))
            sentVectors[parN].append((numpy.ndarray(len(words), dtype=[('ind', numpy.uint16), ('pos', numpy.uint16)])))
            # a "vector" consists of multiple tuples. The first value in a tuple stores the index related to the
            # word, the second one - the position of the word within the sentence. If the word occurs more than once
            # within the same sentence, it occupies more than one entry in the vector so that the positions could be
            # stored. To get the term frequency for a certain word, the number of these entries should be calculated.
            # The entries will be sorted by the indexes associated distinct words.
            sentN += 1
            i = 0
            while i < len(words):
                sentVectors[parN][sentN][i][1] = i
                i += 1
            wordN = -1  # number of words processed in this sentence
            for word in words:
                wordsTotal[parN] += 1
                wordN += 1
                if word in dict:    # if the word was already added in the dictionary
                    tmp = dict[word]
                    parFreq[parN][tmp] += 1
                    sentVectors[parN][sentN][wordN][0] = tmp
                else:
                    dict[word] = wordCount
                    parFreq[parN][wordCount] += 1
                    sentVectors[parN][sentN][wordN][0] = wordCount
                    wordCount += 1
                    if wordCount == MAX_WORDS:
                        print("MAX_WORDS variable is too small. Increase MAX_WORDS so that there is no pair of "
                              "articles in which there are more distinct words than MAX_WORDS")
            sentVectors[parN][sentN] = numpy.sort(sentVectors[parN][sentN], 0, order='ind')  # sorting the tf vector.
    return wordCount


def set_up(a0, a1):
    """
    Get the text of the two articles and set up all the arrays and lists that will be needed later during the alignment.
    These include: creating the dictionary and assigning each word its specific index that will be used elsewhere
    instead of the String itself, filling parFreq, calculating wordsTotal (n of words in every paragraph), creating
    TF vectors (v), creating sInd and sCoor arrays that are used to convert from relative coordinate to absolute
    coordinates in constant time.
    :param a0:  the first article loaded via newselautils.getTokParagraphs
    :param a1:  the second article loaded via newselautils.getTokParagraphs
    :return:    None
    """
    
    global parFreq
    parFreq = (numpy.ndarray((len(a0), MAX_WORDS), numpy.uint16), numpy.ndarray((len(a1), MAX_WORDS), numpy.uint16))
    for par in parFreq[0]:
        par.fill(0)  # zero, because no word appeared yet
    for par in parFreq[1]:
        par.fill(0)

    global wordsTotal
    wordsTotal = (numpy.ndarray(len(a0), numpy.uint32), numpy.ndarray(len(a1), numpy.uint32))
    wordsTotal[0].fill(0)
    wordsTotal[1].fill(0)

    global v
    v = ([], [])
    dict = {}  # the dictionary. Dictionary is only temporary and is not used anywhere else
    wordCount = fill_dictionary(dict, parFreq[0], wordsTotal[0], a0, v[0], 0)
    fill_dictionary(dict, parFreq[1], wordsTotal[1], a1, v[1], wordCount)

    global sInd
    sInd= (numpy.ndarray(len(a0) + 1, numpy.uint16), numpy.ndarray(len(a1) + 1, numpy.uint16))
    sInd[0][0] = 0
    sInd[1][0] = 0
    i = 1
    while i < len(a0) + 1:
        sInd[0][i] = len(a0[i - 1]) + sInd[0][i - 1]
        i += 1
    i = 1
    while i < len(a1) + 1:
        sInd[1][i] = len(a1[i - 1]) + sInd[1][i - 1]
        i += 1

    global sCoor
    sCoor = (numpy.ndarray(sInd[0][len(a0)], numpy.uint16), numpy.ndarray(sInd[1][len(a1)], numpy.uint16))
    i=0
    j=0
    while i < len(sCoor[0]):
        if i == sInd[0][j+1]:
            j += 1
        sCoor[0][i] = j
        i += 1
    i = 0
    j = 0
    while i < len(sCoor[1]):
        if i == sInd[1][j + 1]:
            j += 1
        sCoor[1][i] = j
        i += 1

    global parSim
    parSim = numpy.ndarray((len(a0), len(a1)), numpy.float16)
    parSim.fill(-1)
    global sentSim
    sentSim = numpy.ndarray((sInd[0][len(a0) - 1] + len(a0[len(a0) - 1]),
                            (sInd[1][len(a1) - 1] + len(a1[len(a1) - 1]))), numpy.float16)
    sentSim.fill(-1)


def write_result(slug, loLevel, hiLevel, allparagraphs):
    """
    Write the results of the alignment to the files in the output directory
    :param slug:            the slug that was just processed
    :param loLevel:         the lower one of two levels compared
    :param hiLevel:         the higher one of two levels compared
    :param allparagraphs:   the text of all articles with this slug loaded via newselautils.getTokParagraphs
    :return:                None
    """
    with open(path.OUTDIR_SENTENCES + slug + '-cmp-' + str(loLevel) + '-' + str(hiLevel) + '.csv', 'w') as file:
        # writing all sentence alignments
        file.write(slug + '.en.' + str(loLevel) + '\t\t' + slug + '.en.' + str(hiLevel) + '\tFirst line contains '
                    'the list, in which for each paragraph in the first article is given a number of sentences that '
                    'occurred before this paragraph. The second line contains the same array for the second article\n')
        for i in range(len(sInd[0])):
            file.write(str(sInd[0][i])+" ")
        file.write("\n")
        for i in range(len(sInd[1])):
            file.write(str(sInd[1][i])+" ")
        file.write("\n")
        for block in result:  # for all blocks of alignments. A block of alignments is a set of alignment that share
            # sentences between them
            for i in range(len(block)-1):
                file.write(str(block[i][0][0]+1)+':'+str(block[i][0][1]+1) + ',' +
                           str(block[i][1][0]+1)+':'+str(block[i][1][1]+1) + '\t')
            file.write(str(block[-1][0][0] + 1) + ':' + str(block[-1][0][1] + 1) + ',' +
                       str(block[-1][1][0] + 1) + ':' + str(block[-1][1][1] + 1) + '\n')

    with open(path.OUTDIR_PARAGRAPHS + slug + '-cmp-' + str(loLevel) + '-' + str(hiLevel) + '.csv', 'w') as file:
        # writing all the paragraph alignments
        file.write(slug + '.en.' + str(loLevel) + '\t\t' + slug + '.en.' + str(hiLevel) + '\tFirst line contains '
        'the overall number of paragraphs in the first and second articles \n'+str(len(allparagraphs[loLevel]))+' '+
                                                                               str(len(allparagraphs[hiLevel]))+ '\n')
        while i < len(parResult):
            # all sentences from the first article
            j = 0
            while j < len(parResult[i][0]) - 1:
                file.write(str(parResult[i][0][j] + 1) + ',')
                j += 1
            file.write(str(parResult[i][0][j] + 1) + '\t\t')
            j = 0
            # all sentences from the second article
            while j < len(parResult[i][1]) - 1:
                file.write(str(parResult[i][1][j] + 1) + ',')
                j += 1
            file.write(str(parResult[i][1][j] + 1) + '\n')
            i += 1


def extract_results():
    """
    converts the results of the paragraphs' alignment from the matrix to a list (from parResultMatrix to parResult)
    :return: None
    """
    global parResult
    parResult = []
    for par0 in range(len(parResultMatrix)):
        for par1 in range(len(parResultMatrix[par0])):
            if parResultMatrix[par0][par1]:
                nextAlignment = ([par0], [par1])
                parResultMatrix[par0][par1] = False
                checkFirst = queue.Queue()
                checkSecond = queue.Queue()
                checkFirst.put(par0)
                checkSecond.put(par1)
                while (not checkFirst.empty())or(not checkSecond.empty()):
                    if not checkFirst.empty():
                        p0 = checkFirst.get()
                        for p1 in range(len(parResultMatrix[p0])):
                            if parResultMatrix[p0][p1]:
                                parResultMatrix[p0][p1] = False
                                if not p1 in nextAlignment[1]:
                                    nextAlignment[1].append(p1)
                                checkSecond.put(p1)
                    if not checkSecond.empty():
                        p1 = checkSecond.get()
                        for p0 in range(len(parResultMatrix)):
                            if parResultMatrix[p0][p1]:
                                parResultMatrix[p0][p1] = False
                                if not p0 in nextAlignment[0]:
                                    nextAlignment[0].append(p0)
                                checkFirst.put(p0)
                parResult.append(nextAlignment)


def sim_in_articles(slug, paragraphs, levels):
    """
    Pairwise compare the levels (given by the levels parameter) of the article, given by paragraphs - the list of
    the tokenized articles with this slug (obtained from newselautils.getTokParagraphs)
    :param slug: slug to process
    :param paragraphs: the list of the tokenized articles with this slug (obtained from newselautils.getTokParagraphs)
    :param levels: the same as levels parameter in align_all and align_particular.
    :return: None
    """
    #   for levels except last, starting from simplest
    for comp in levels:
        if comp[1] >= len(paragraphs):
            continue  # if the article was not adapted for this level
        # print('Matching levels %d and %d' % (comp[0], comp[1]))
        set_up(paragraphs[comp[0]], paragraphs[comp[1]])
        global result  # cleaning the result variables that are filled with results of previous alignments
        global parSim
        result = []
        global parResultMatrix
        parResultMatrix = numpy.ndarray((len(paragraphs[comp[0]]), len(paragraphs[comp[1]])), numpy.bool)
        parResultMatrix.fill(False)
        align_paragraphs(len(paragraphs[comp[0]]), len(paragraphs[comp[1]]))
        for i in range(comp[2]-1):
            for par in parSim:  # resetting parSim before calling align_paragraphs for the second (third) time
                par.fill(-1)
            align_paragraphs(len(paragraphs[comp[0]]), len(paragraphs[comp[1]]))
        extract_results()
        write_result(slug, comp[0], comp[1], paragraphs)


def align_first_n(nToAlign = -1, levels = [(0, 1, 3), (1, 2, 3), (2, 3, 2), (3, 4, 2), (4, 5, 2)]):
    """
    Create alignments for the first nToAlign slugs. If nToAlign=-1, align all slugs.
    :param nToAlign: the number of slugs to align. If nToAlign = -1, all the slugs will be aligned
    :param levels: which levels to align and how many times to run the algorithm. For example, align_first_n(-1,[(0,1,2)])
    will align all 0-1 articles and will run the algorithm twice for each alignment.
    align_first_n(-1,[(0,1,1),(1,2,1),(2,3,1),(3,4,1),(4,5,1)]) will align all adjacent levels running the algorithm once
    for every level. The levels parameter should be a list of tuples of three elements. The first element is the lower
    level to align, the second is the higher level to align, the third is how many times to run the algorithm for this
    pair of levels.
    :return: None
    """
    info = loadMetafile()
    eu.calculate(MAXIMUM_PARAGRAPHS, MAXIMUM_PARAGRAPHS, VICINITIES, SENTENCE_VICINITIES) # one-time operation that will 
    # later allow to iterate over the matrix by increasing the euclidean distance from a specific entry
    nSlugs = 0
    i = 0
    for comparison in levels:
        if comparison[0] >= comparison[1]:
            print("the lower level should be indicated first")
            return
    while (i < len(info))and((nToAlign == -1)or(nSlugs < nToAlign)):
        artLow = i  # first article with this slug
       	slug = info[i]['slug']
        nSlugs += 1
        if nToAlign == -1:
            print("Processing slug... "+ slug+' '+str(round(i/float(len(info)) * 100, 3))+'% of the task completed')
        else:
            print("Processing slug... "+ slug+' '+str(round(nSlugs/float(nToAlign) * 100, 3))+'% of the task completed')
        while i < len(info) and slug == info[i]['slug']:
            i += 1
        artHi = i  # one more than the number of the highest article with this slug
        sim_in_articles(slug, list(map(getTokParagraphs, info[artLow:artHi])), levels)  # the articles in the metafile
        #  should be ordered by the slug and then by increasing the level of adaptation


def align_particular(slugs, levels=[(0, 1, 3), (1, 2, 3), (2, 3, 2), (3, 4, 2), (4, 5, 2)]):
    """
    Create alignments for the slugs that are indicated by the slugs parameter.
    :param slugs: the list of slugs to process
    :param levels: which levels to align and how many times to run the algorithm. For example,
    align_particular(["standford_did_smth"], [(0,1,2)]) will align the versions of the "standford_did_smth" article
    with levels 0 and 1 and will run the algorithm twice for each alignment.
    align_particular(["standford_did_smth"], [(0,1,1),(1,2,1),(2,3,1),(3,4,1),(4,5,1)]) will align all adjacent levels
    running the algorithm once for every level. The levels parameter should be a list of tuples of three elements.
    The first element is the lower level to align, the second is the higher level to align, the third is how many times
    to run the algorithm for this pair of levels.
    :return: None
    """
    info = loadMetafile()
    eu.calculate(MAXIMUM_PARAGRAPHS, MAXIMUM_PARAGRAPHS, VICINITIES, SENTENCE_VICINITIES) # one-time operation that will
    # later allow to iterate over the matrix by increasing the euclidean distance from a specific entry
    for comparison in levels:
        if comparison[0] >= comparison[1]:
            print("the lower level should be indicated first")
            return
    for slug in slugs:
        # print("Processing slug... " + slug)
        artLow = autils.get_lowest_element_with_slug(slug, info)
        artHi = artLow
        while artHi < len(info) and slug == info[artHi]['slug']:
            artHi += 1
        sim_in_articles(slug, list(map(getTokParagraphs, info[artLow:artHi])), levels)


if __name__ == "__main__":
    """ align_particular(["10dollarbill-woman", "ski-swat", "ancient-astronomy", "angrybirds-spying", "slavery-reparations",
         "pharaoh-tomb", "agtech-food", "vertical-gardens", "turkey-riots", "aztec-discovery", "boston-timecapsule",
          "africa-lions"])"""
    # align_first_n()
    # align_particular(["boston-timecapsule", "aztec-discovery", "fed-dollars", "africa-lions",  "haitian-migrants"])
    # align_particular(["cat-apathy"],[(0,3,2)])
    align_first_n()
