# Contains all the path variables that need to be configured before running
# the program on a new computer

import sys

USERDIR = '/home/af9562'
if sys.platform == 'darwin':
    USERDIR = '/Users/alexanderfedchin'
BASEDIR = USERDIR + '/newsela'
METAFILE = BASEDIR + '/articles_metadata.csv'
PARSERDIR = BASEDIR + '/stanford-parser-full-2015-12-09/'
OUTDIR_SENTENCES = BASEDIR+'/output/sentences/'
OUTDIR_PARAGRAPHS = BASEDIR+'/output/paragraphs/'
OUTDIR_NGRAMS = BASEDIR+'/output/ngrams/'
OUTDIR_PRECALCULATED = OUTDIR_NGRAMS+'ngramsByFile/'
OUTDIR_TO_DELETE = OUTDIR_NGRAMS+'toDelete/'
OUTDIR_TOK_NGRAMS = OUTDIR_NGRAMS+'tokenizedForNgrams/'
OUTDIR_PERPLEX = OUTDIR_NGRAMS+'perplexity/'
MANUAL_SENTENCES = BASEDIR+'/manual/sentences/new_format/'
MANUAL_PARAGRAPHS = BASEDIR+'/manual/paragraphs/'

PARSERPROG = 'custom/Parser'
TOKENIZERPROG = 'custom/Tokenizer'

CLASSPATH = ':'.join(['.',PARSERDIR,PARSERDIR + 'stanford-parser.jar', PARSERDIR + 'stanford-parser-3.6.0-models.jar', PARSERDIR + 'slf4j-api.jar'])

