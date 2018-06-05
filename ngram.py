" this module is a tool for automatic creation of n-grams for newsela data "


#TODO Get Tok Paragraphs
import newselautil as nutils
import classpaths as path
import subprocess
import sys
import io
import os.path
is_py2 = sys.version[0] == '2'
if is_py2:
    import Queue as queue
else:
    import queue as queue

MERGE_PER_TIME = 200 # the maximum amount of ngram-files to merge in one call. Developers of SRLIM suggest that
# it is better to merge fewer files at a time (file: doc/intro-lm). There is also a limit for input
# parameters for one call, which does not allow to merge all articles at once

MINUS_INFINITY = '-1000000'  # the value that will be assigned to the log of the probability if the probability is zero


def build_ngrams(outputFile, nToProcess=-1, levels=[0,1,2,3,4,5], mingrade=0, maxgrade=12, exclude = None,
                 onlyEnglish = True, usePrecalculated = True):
    """
    Count ngrams for nToProcess slugs (and indicated levels) (put them into path.OUTPUT_NGRAMS/ngrams_by_file/) and
    then merge them all into one file. Then create a language model using Kneser-Nay
    smoothing. All the files in path.OUTPUT_NGRAMS/toDelete/ (but not the folder itself) can be deleted after the
    program finishes.
    :param outputFile: the name of the files that will contain the merged output. outputFile.ngrams will contain the
    ngrams and output.bo - the language model
    :param nToProcess: the number of slugs to process. If nToProcess = -1, all the slugs will be processed
    :param levels: the list of levels for which to build the models
    :param mingrade: the lowest grade for which to calculate the ngrams
    :param maxgrade: the highest grade for which to calculate the ngarms
    :param exclude: the name of the file that contains the list of all articles that should be excluded (the first
    parameter on each line (except for the first line) should be the name of a 0 level article). The names should go in
    the same order as they are in metafile (that is in alphabetical order)
    :param onlyEnglish: If True, only english articles will be processed
    :param usePrecalculated: If True, the program will not reevaluate the ngrams for the files which are already
    in the ngrams_by_file folder
    :return: None
    """
    info = nutils.loadMetafile()
    nSlugs = 0  # this will store the number of the slug that is currently processed

    q = queue.Queue()  # the program first creates n-gram models for all the articles separately, then merges them all
    # together and only afterwards creates the language model. The queue will store the names of all files that are to
    # be merged

    if (nToProcess == 0) or (nToProcess < -1):
        print("nToProcess parameter takes either a positive value or -1")
        return
    if len(levels) < 1:
        print("at list one level should be indictaed")
        return
    if (mingrade > maxgrade):
        tmp = mingrade
        mingrade = maxgrade
        maxgrade = tmp

    if exclude is not None:
        with io.open(exclude) as file:
            excluded = file.readlines()
            for i in range(len(excluded)):
                excluded[i] = excluded[i].split(" ")
        e_index = 1  # the position of the next article in excluded array that should be taken into account
    else:
        excluded = []
        e_index = 1

    i = 0
    while (i < len(info))and((nToProcess == -1)or(nSlugs < nToProcess)):
        artLow = i  # first article with this slug
        slug = info[i]['slug']
        while i < len(info) and slug == info[i]['slug']:
            i += 1
        artHi = i  # one more than the number of the highest article with this slug
        if (e_index >= len(excluded))or(info[artLow]["filename"] != excluded[e_index][0]+".txt"):  # if this article
            # should not be excluded from the model
            nSlugs += 1
            if (info[artLow]["language"] == "en") or not onlyEnglish:
                for level in levels:  # the level of adaptation (i.e. 0 - is the original etc.)

                    if level >= artHi - artLow:
                        continue
                    grade = float(info[artLow + level]["grade_level"])  # the grade level (not the level of adaptation
                    if (grade < mingrade) or (grade > maxgrade):
                        continue

                    q.put(path.OUTDIR_PRECALCULATED + info[artLow + level]["filename"] + ".ngrams")
                    if usePrecalculated and os.path.isfile(
                                            path.OUTDIR_PRECALCULATED + info[artLow + level]["filename"] + ".ngrams"):
                        continue

                    subprocess.call(["ngram-count", "-text", path.OUTDIR_TOK_NGRAMS +
                                     info[artLow + level]["filename"] + ".tok", "-sort", "-write",
                                     path.OUTDIR_PRECALCULATED + info[artLow + level]["filename"] + ".ngrams"])
                    # crete the n-gram model for this particular file

                if nToProcess == -1:
                    print(
                        "Processing slug... " + slug + ' ' + str(round(i / float(len(info)) * 100, 3)) + '% completed')
                else:
                    print("Processing slug... " + slug + ' ' + str(
                        round(nSlugs / float(nToProcess) * 100, 3)) + '% completed')
        else:
            e_index += 1

    extraFilesCount = 0  # the number of extra files that could be deleted after the completion of the program
    # (these are temporaraly merges and will be located in path.OUTPUT_NGRAMS/toDelete/)

    while q.qsize() > MERGE_PER_TIME:  # align MERGE_PER_TIME files per call
        next_input = ["ngram-merge"]
        print("Files left to merge:"+str(q.qsize()))
        for i in range (MERGE_PER_TIME):
            next_input.append(q.get())
        with open(path.OUTDIR_TO_DELETE+str(extraFilesCount)+".ngrams", 'w') as file:
            if is_py2:
                file.write(subprocess.check_output(next_input, shell=False))
            else:
                file.write(subprocess.run(next_input, stdout=subprocess.PIPE).stdout.decode('utf-8'))
        q.put(path.OUTDIR_TO_DELETE+str(extraFilesCount)+".ngrams")
        extraFilesCount += 1

    next_input = ["ngram-merge"]
    for i in range (q.qsize()):
        next_input.append(q.get())
    with open(path.OUTDIR_NGRAMS+outputFile+".ngrams", 'w') as file:
        if is_py2:
            file.write(subprocess.check_output(next_input, shell=False))
        else:
            file.write(subprocess.run(next_input, stdout=subprocess.PIPE).stdout.decode('utf-8'))  # this will create
        # the outputFile.ngrams that will contain all the merged ngrams

    subprocess.call(["ngram-count", "-read", path.OUTDIR_NGRAMS+outputFile+".ngrams",
                     "-lm", path.OUTDIR_NGRAMS+outputFile+".bo", "-kndiscount"])


def delete_pars_symbols():
    """Read all tokenized articles and delete @PGPH lines because they contaminate ngrams. Also, replace ## tags with
    @TITLE tags, because otherwise they are not processed by SRILM"""
    info = nutils.loadMetafile()
    i = 0
    while i < len(info):
        artLow = i  # first article with this slug
        slug = info[i]['slug']
        while i < len(info) and slug == info[i]['slug']:
            i += 1
        artHi = i  # one more than the number of the highest article with this slug
        for level in range(artHi-artLow):
            with io.open(path.BASEDIR + "/articles/" + info[artLow + level]["filename"] + ".tok") as file:
                lines = file.readlines()
            with io.open(path.OUTDIR_TOK_NGRAMS + info[artLow + level]["filename"] + ".tok", 'w') as file:
                for line in lines:
                    splitted = line.split()
                    if splitted[0] == '@PGPH':
                        continue
                    if splitted[0] == '##' or splitted[0] == '###':
                        file.write('@TITLE'+line[2:])
                    else:
                        file.write(line)


def article_perplexity(articleName,lmName):
    """
    Calculate the probability of each word in an article according to the designated language model. Write all of this
    to a file in perplexity folder. The first line contains the overall perplexity of the article as given by SRILM:
    "Perplexity is given with two different normalizations: counting all input tokens and excluding end-of-sentence tags"
    Both of these values are on the first line. Every one of the following lines represents a sentence and contains the
    probabilities for every word in this sentence given as the logarithms (base 10). If the probability is equal, to
    MINUS_INFINITY value, the word is out of vocabulary.
    :param articleName: the name of the article to calculate perplexity
    :param lmName: the name of the language model (.bo extension), found in OUTDIR_NGRAMS.
    :return: the perplexity (as a tuple)
    """
    if is_py2:
        ouput=subprocess.check_output(["ngram", "-lm", path.OUTDIR_NGRAMS + lmName, "-ppl", path.OUTDIR_TOK_NGRAMS +
                                articleName, "-debug", "2"], shell=False).split('\n')
    else:
        ouput = subprocess.run(["ngram", "-lm", path.OUTDIR_NGRAMS + lmName, "-ppl", path.OUTDIR_TOK_NGRAMS +
                                articleName, "-debug", "2"], stdout=subprocess.PIPE).stdout.decode('utf-8').split('\n')
    with io.open(path.OUTDIR_TOK_NGRAMS + articleName) as file:
        lines = file.readlines()
        for i in range(len(lines)):
            lines[i] = lines[i].split(' ')
    i = 1
    with open(path.OUTDIR_PERPLEX+articleName+".prob", 'w') as file:
        perplexity = ouput[-2].split(' ')
        file.write(str(perplexity[-3]) + " " + str(perplexity[-1])+"\n")
        for sent in range(len(lines)):
            for word in range(len(lines[sent])):
                value = ouput[i].split('\t=')[1].split(' ')[4]
                if value == '-inf':  # replace -inf with MINUS_INFINITY
                    file.write(MINUS_INFINITY+' ')
                else:
                    file.write(value+' ')
                i += 1
            value = ouput[i].split('\t=')[1].split(' ')[4]
            if value == '-inf':  # replace -inf with MINUS_INFINITY
                file.write(MINUS_INFINITY+'\n')
            else:
                file.write(value+'\n')
            i += 5
    return float(perplexity[-3]), float(perplexity[-1])

def calculate_all_perplexities(lmName, nToProcess=-1, levels=[0,1,2,3,4,5], mingrade=0, maxgrade=12, include = None,
                 onlyEnglish = True):
    """
    calculate perplexities for articles designated by adaptation level, grade, language or slug
    :param lmName: the name of the language model (.bo extension), found in OUTDIR_NGRAMS.
    :param nToProcess: number of slugs to calculate perplexities for
    :param levels: the list of levels for which to calculate perplexities
    :param mingrade: the lowest grade for which to calculate perplexities
    :param maxgrade: the highest grade for which to calculate perplexities
    :param include:  the name of the file that contains the list of all articles for which to calculate perplexities
    (the first parameter on each line (except for the first line) should be the name of a 0 level article).
    The names should go in the same order as they are in metafile (that is in alphabetical order). If include = NONE,
    the perplexities will be calculated for the first nToProcess slugs
    :param onlyEnglish: If True, only english articles will be processed
    :return:
    """
    info = nutils.loadMetafile()
    nSlugs = 0  # this will store the number of the slug that is currently processed

    if (nToProcess == 0) or (nToProcess < -1):
        print("nToProcess parameter takes either a positive value or -1")
        return
    if len(levels) < 1:
        print("at list one level should be indictaed")
        return
    if (mingrade > maxgrade):
        tmp = mingrade
        mingrade = maxgrade
        maxgrade = tmp

    if include is not None:
        with io.open(include) as file:
            included = file.readlines()
            for i in range(len(included)):
                included[i] = included[i].split(" ")
        i_index = 1  # the position of the next article in included array that should be taken into account
    else:
        included = []
        i_index = 1

    average_perpl = (.0,.0)
    num_of_files = .0

    i = 0
    while (i < len(info)) and ((nToProcess == -1) or (nSlugs < nToProcess)):
        artLow = i  # first article with this slug
        slug = info[i]['slug']
        while i < len(info) and slug == info[i]['slug']:
            i += 1
        artHi = i  # one more than the number of the highest article with this slug
        if (i_index >= len(included)) or (info[artLow]["filename"] != included[i_index][0] + ".txt"):  # if the
            # perplexities for this article should not be calculated
            continue
        i_index +=1
        if (info[artLow]["language"] == "en") or not onlyEnglish:
            for level in levels:  # the level of adaptation (i.e. 0 - is the original etc.)

                if level >= artHi - artLow:
                    continue
                grade = float(info[artLow + level]["grade_level"])  # the grade level (not the level of adaptation
                if (grade < mingrade) or (grade > maxgrade):
                    continue

                curr_perpl = article_perplexity(info[artLow + level]["filename"] + ".tok", lmName)
                num_of_files += 1
                average_perpl=(average_perpl[0]+curr_perpl[0], average_perpl[1]+curr_perpl[1])

            if nToProcess == -1:
                print(
                    "Processing slug... " + slug + ' ' + str(round(i / float(len(info)) * 100, 3)) + '% completed')
            else:
                print("Processing slug... " + slug + ' ' + str(
                    round(nSlugs / float(nToProcess) * 100, 3)) + '% completed')
    print(str(average_perpl[0]/num_of_files)+ " "+ str(average_perpl[1]/num_of_files))

if __name__ == "__main__":
    delete_pars_symbols()
    build_ngrams("test", levels=[1,2,3,4], mingrade=2, maxgrade=9, exclude=path.BASEDIR+"/NewselaSimple03test.idx")
    calculate_all_perplexities("test.bo", levels=[0], include=path.BASEDIR+"/NewselaSimple03test.idx")
