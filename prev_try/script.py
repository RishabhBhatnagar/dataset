from pandas import read_csv
from os.path import isfile
from pandas import DataFrame
from nltk import sent_tokenize
from nltk import word_tokenize
from nltk import pos_tag
from collections import Counter
from time import time
from itertools import count
import re
import enchant

# Next two just for showing file not found error 
import errno
from os import strerror


def load_data(file_path, encoding):
    """
    Loads the data from given file path and with given encoding using LBYL
    """
    data: DataFrame
    
    # checking if filename is not empty or none and that file exists.
    if file_path and isfile(file_path):
    
        # The existing file is a csv or a tsv
        if file_path[-3:] == 'tsv':
            data = read_csv(file_path, sep='\t', encoding=encoding)
        elif file_path[-3:] == 'csv':
            data = read_csv(file_path, sep=', ', encoding=encoding)
        else:
            # file path given is not of a csv or a tsv file.
            raise TypeError("Data file should either be csv or tsv")
    else:
        # There was no file with given path
        raise FileNotFoundError(errno.ENOENT, strerror(errno.ENOENT), file_path)
    
    return data


def punctuation_replace(string, replace_with=' ', punctuations=',/*-()[]\\!'):
    for char in punctuations:
        string = string.replace(char, replace_with)
    return string


def main(file_path, encoding):
    KEY_ESSAY = 'essay'
    KEY_SCORE_1 = 'rater1_domain1'
    KEY_SCORE_2 = 'rater2_domain1'
    
    mKEY_SCORE = 'score'             # This is not in dataset.
    mKEY_WORD_COUNT = 'word_count'
    mKEY_SENTENCE_COUNT = 'sentence_count'
    mKEY_SPELLING_MISTAKES = 'spelling_mistakes_count'
    
    dictionary = enchant.Dict('en_US')
    df = load_data(file_path, encoding=encoding)
    
    if True:
        ## LBYL check for all keys.
        
        vars_snap = locals()                 # temp copy of all local variables.    
        # Finding all variable whose identifier starts with 'KEY_'
        keys = [vars_snap[var] for var in vars_snap if var.startswith('KEY_')]
        for key in keys:
            if key not in df:
                raise KeyError(key)
    
    # Truncating all other unwanted features.
    df = df[[KEY_ESSAY, KEY_SCORE_1, KEY_SCORE_2]]
    
    # Removing all rows with any of the field as None/NaN
    df.dropna()
    
    # Merging both  the scores.
    df[mKEY_SCORE] = df[KEY_SCORE_1] + df[KEY_SCORE_2]
    
    # Keeping dataset with only two columns viz: essay and score
    df = df[[KEY_ESSAY, mKEY_SCORE]]
    
    # counting the number of words in each essay.
    df[mKEY_WORD_COUNT] = df[KEY_ESSAY].apply(lambda x:len(x.split()))
    
    # counting the number of sentences in each essay.
    df[mKEY_SENTENCE_COUNT] = df[KEY_ESSAY].apply(lambda x:len(x.split('.')))
    
    # removing all the words starting with @ symbol.
    df[KEY_ESSAY] = df[KEY_ESSAY].apply(lambda s: re.sub(r'[^\x00-\x7F]+', ' ', re.sub('\s*@\w*\d*\s*', ' ', s)))
    
    if True:
        print("Counting number of pos.")
        
        wanted = ['NN', 'NNS', 'NNP', "JJ", 'JJR', 'JJS', 'RB', 'RBR', 'RBS', 'VBG', 'VBN', 'VBP', 'VBZ']
        # counting the number of POS.
        dicts_df = DataFrame(dict(item for item in Counter(map(lambda x:x[1], pos_tag(x.split()))).items() if item[0] in wanted) for x in df[KEY_ESSAY])
        
        # Replacing all NaN by 0
        dicts_df.fillna(0, inplace=True)
        
        # This is for me so that i can use get method which won't give keyerror.
        dicts = dicts_df.to_dict()
        
        wanted = [key for key in wanted if dicts.get(key) is not None]
        df['JJ'] = dicts_df[[key for key in ["JJ", 'JJR', 'JJS'] if key in wanted]].sum(axis='columns')
        df['NN'] = dicts_df[[key for key in ['NN', 'NNS', 'NNP'] if key in wanted]].sum(axis='columns')
        df['RB'] = dicts_df[[key for key in ['RB', 'RBR', 'RBS'] if key in wanted]].sum(axis='columns')
        df['VB'] = dicts_df[[key for key in ['VBG', 'VBN', 'VBP', 'VBZ'] if key in wanted]].sum(axis='columns')
    if True:
        print("spelling check")
        # Checking the number of spelling mistakes
        df[mKEY_SPELLING_MISTAKES] = df[KEY_ESSAY].apply(lambda x: len(['' for word in punctuation_replace(x).split() if not dictionary.check(word) ]))
        
    df.to_csv('train_extracted_feature.csv', index=False)


if __name__ == '__main__':
    tsv_name = 'training.tsv'
    encoding = 'latin'
    main(file_path=tsv_name, encoding=encoding)
