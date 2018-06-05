# Run stanford parser ParseDemo on utf-8 text file.
# Be sure to compile ~/lib/standford-parser*/custom/Parser

import subprocess
import sys

import classpaths as path


def parse(textfile):
    '''Run parser and return all output as a string.'''
    #subprocess.call(['java','-cp',CLASSPATH,PROG,MODELS,textfile],shell=False)
    #    output = subprocess.check_output(['java','-cp',CLASSPATH,'-Xmx8000m',,PROG,MODELS,textfile],shell=False)
    output = subprocess.check_output(['java','-cp',path.CLASSPATH,'-Xmx8000m',path.PARSERPROG,path.MODELS,textfile],shell=False)
    return output

def tokenize(textfile):
    '''Run tokenizer.  Output in textfile.tok'''
    output = subprocess.check_output(['java','-cp',path.CLASSPATH,path.TOKENIZERPROG,textfile],shell=False)


def main():
    textfile = sys.argv[1]
    print (parse(textfile))
    #tokenize(textfile)

        
if __name__ == "__main__":
    main()
