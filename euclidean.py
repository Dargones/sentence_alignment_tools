"""
This module allows quick iteration over increasing euclidean distance from any given point in a given matrix. Euclidean
distance is calculated from the positions of the elements within the matrix. While iterating over the matrix, the
algorithm ONLY CONSIDERS THOSE ELEMENTS, WHOSE X and Y COORDINATES ARE GREATER THAN THOSE OF THE ORIGIN FROM WHICH
THE ITERATION STARTS

calculate(int n, int m) - this method ensures that the module will work adequately with a matrix of the size n*m. If the
requested size of teh matrix is larger than that calculated previously, the module will resize the _euclidean array.
This procedure should work in nm*log(nm) time.

closest((uint,uint) start, uint startIndex, uint len1, uint len2, function, extraParameters=[]) - Performs a
search for a point in the matrix that satisfies the expression. The search is performed by increasing the euclidean
distance from a given point. Returns the first pair of coordinates for which the expression is evaluated as true one,
or None if there are no such coordinates

update_vicinities(vicinities, boolean isForParagraphs) - sets the values for parStart
and sentStart. Updates the vicinities list if there is no way to set a unique parStart and SentStart

main() - only for debugging
"""

import numpy
import math

_euclidean = None  # the array of coordinates. The coordinates are stored in the order of increasing euclidean distance
# from point (0.0). All coordinates are positive
_N = 0  # width of the matrix for which the array was created last time
_M = 0  # height of the matrix
parStart = 0  # since the algorithm in align.py at first searches for the paragraphs in VICINITIES and only uses
# euclidean distance afterwards, there is no need to include the point covered by vicinities in the euclidean array.
# Only the elements of the array starting from PAR_START should be checked.
sentStart = 0  # same for sentences


def calculate(n, m, parVicinities, sentVicinities):
    """
    If the array of the requested size already exists, does nothing.
    Otherwise creates the new array for a matrix of a given size (m*n). Then fills this array with points from this
    matrix by increasing the sum of coordinates (going by the diagonal from top right to low left boundaries).
    The array should be partially sorted after this procedure. Then quickSort is used for sorting the array
    by the euclidean distance from the origin. Finally, the algorithm determines the values of parStart and sentStart
    and appends extra points to parVicinities or sentVicinities if it is necessary for setting unique parStart and
    sentStart values
    :param n:               - width of the matrix (int)
    :param m:               - height of the matrix (int)
    :param parVicinities:   - VICINITIES list from align.py
    :param sentVicinities:  - SENTENCE_VICINITIES list from align.py
    :return:   - None
    """
    global _N
    global _M
    global _euclidean
    if (n <= _N) and (m <= _M):
        return
    _N = n
    _M = m
    _euclidean = numpy.ndarray(n * m,
                               dtype=[('x', numpy.uint16), ('y', numpy.uint16), ('euclidean', numpy.float16)])
    count = 0  # the next position to be filled in the array
    dist = 0  # iterating by the sum of the coordinates starting from the sum=0
    next = (0, dist - 1)  # next = (0, -1) so that in the first iteration of the while loop, the first "if" will be True
    while dist < n + m - 1:
        if next[0] + next[1] < dist:  # dist was updated and it is necessary to set up next
            if dist >= n:
                next = (n - 1, dist - n + 1)
            else:
                next = (dist, 0)
        else:
            _euclidean[count]['x'] = next[0]
            _euclidean[count]['y'] = next[1]
            _euclidean[count]['euclidean'] = math.sqrt(next[0] * next[0] + next[1] * next[1])
            count += 1
            next = (next[0] - 1, next[1] + 1)  # going by the diagonal from the top right to the low left boundaries
            if (next[1] >= m) or (next[0] < 0):
                dist += 1
    _euclidean=numpy.sort(_euclidean, 0, order='euclidean')
    _update_vicinities(parVicinities, True)  # determines the value of parStart
    _update_vicinities(sentVicinities, False)  # determines the value of sentStart


def closest(start, startIndex, len1, len2, function, extraParameters=[]):
    """
    Performs a search for a point in the matrix that satisfies the expression. The search is performed by increasing the
    euclidean distance from a given point. Returns the first pair of coordinates for which the expression is 
    evaluated as true one, or None if there are no such coordinates
    :param start:           - the point from which to start the search (tuple of uints)
    :param startIndex:      - the index in euclidean array from which to start the search (unit)
    :param len1:            - length of the first axis of the matrix (uint)
    :param len2:            - length of the second one (uint)
    :param function:        - an expression to evaluate (function that returns a boolean and takes at least two
                            parameters: the starting point and the distance from it)
    :param extraParameters: - extra parameters to pass to the function if needed
    :return: the coordinates (relative to start) that first satisfy the function. Returns None if there are no such
                                                                                                        coordinates
    """
    change0 = len1 - start[0]  # the length of the first axis of the matrix that is searched in (as if start coordinate
    # was the origin)
    change1 = len2 - start[1]  # same for the second axis
    maxDistance = change0 + change1
    i = startIndex
    while (i < len(_euclidean))and(_euclidean[i][0] + _euclidean[i][1] < maxDistance-1):
        if (_euclidean[i][0] < change0) and (_euclidean[i][1] < change1):
            if len(extraParameters) == 0:
                if function(start, _euclidean[i]):
                    return _euclidean[i]
            else:
                if function(start, _euclidean[i], extraParameters):
                    return _euclidean[i]
        i += 1
    return None


def _update_vicinities(vicinities, isForParagraphs):
    """
    sets the values for parStart and sentStart. Updates the vicinities list if there is no way to set a unique
    parStart and SentStart
    :param vicinities:      - either VICINITIES or SENTENCE_VICINITIES lists from align.py
    :param isForParagraphs: - true if vicinities=VICINITIES, false if vicinities=SENTENCE_VICINITIES
    :return:                - None
    """
    max = -1.0  # the maximum euclidean distance between the origin and a point in Vicinities
    maxInd = -1  # the index of this point (or the maximum index among such points if there are many of those)
    for vicinity in vicinities:
        for point in vicinity:
            if point[0] * point[0] + point[1] * point[1] >= max:  # if the new maximum is found
                i = 0  # then search for this point in euclidean
                while (i < len(_euclidean)) and ((_euclidean[i][0] != point[0]) or (_euclidean[i][1] != point[1])):
                    i += 1
                if i > maxInd:
                    maxInd = i
                    max = point[0] * point[0] + point[1] * point[1]
    # for every element in euclidean which is at the position less than max and which is not in vicinities - add it to
    # vicinities
    i = 0
    while i < maxInd:
        foundInVicinities = False
        for vicinity in vicinities:
            for point in vicinity:
                if (_euclidean[i][0] == point[0]) and (_euclidean[i][1] == point[1]):
                    foundInVicinities = True
                    break
        if (not foundInVicinities)and((_euclidean[i][0] != 0)or(_euclidean[i][1] != 0)): # starting point itself should
            # not be added to vicinities
            vicinities.append(((_euclidean[i][0], _euclidean[i][1]),))
        i += 1
    # update parStart and sentStart
    if isForParagraphs:
        global parStart
        parStart = maxInd
    else:
        global sentStart
        sentStart = maxInd


def main():
    """only for the purposes of debugging"""
    size1 = 5
    size2 = 4
    vicinity1 = [((0, 1),), ((1, 1),)]
    vicinity2 = [((1, 1),), ((1, 2), (2, 1))]
    calculate(size1, size2, vicinity1, vicinity2)
    print("arranging the points in " + str(size1) + "x" + str(size2) + " matrix by euclidean order and updating "
                                                                       "vicinities:\n" + str(
        vicinity1) + " and \n" + str(vicinity2) + ":\n")
    print("the array: " + str(_euclidean) + "/n first vicinity: " + str(vicinity1) + "\n second vicinity" + str(
        vicinity2))


if __name__ == "__main__":
    main()
