import os
import string
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from operator import itemgetter
from collections import Counter
import random
from tkinter import ttk 
from tkinter import *
import regex

END_OF_KURAL = "."
flatten_list = lambda list: [item for sublist in list for item in sublist]
def frequency_of_occurrence(words, specific_words=None):
    """
    Returns a list of (instance, count) sorted in total order and then from most to least common
    Along with the count/frequency of each of those words as a tuple
    If specific_words list is present then SUM of frequencies of specific_words is returned 
    """
    freq = sorted(sorted(Counter(words).items(), key=itemgetter(0)), key=itemgetter(1), reverse=True)
    if not specific_words or specific_words==None:
        return freq
    else:
        frequencies = 0
        for (inst, count) in freq:
            if inst in specific_words:
                frequencies += count        
        return float(frequencies)
        
def has_required_percentage_of_occurrence(words, specific_words=None,required_percent_of_occurrence=0.99):
    actual_percent_of_occurrence = percentage_of_occurrence(words, specific_words=specific_words)
    percent_check = actual_percent_of_occurrence >= required_percent_of_occurrence
    return [percent_check, actual_percent_of_occurrence]

def percentage_of_occurrence(words, specific_words=None):
    """
    Returns a list of (instance, count) sorted in total order and then from most to least common
    Along with the percent of each of those words as a tuple
    If specific_words list is present then SUM of percentages of specific_words is returned 
    """
    frequencies = frequency_of_occurrence(words) # Dont add specific_word as argument here float error happens
    perc = [(instance, count / len(words)) for instance, count in frequencies]
    if not specific_words or specific_words==None:
        return perc
    else:
        percentages = 0
        for (inst, per) in perc:
            if inst in specific_words:
                percentages += per        
        return percentages

class Thirukural:
    chapter_max = 133
    verse_max = 10
    kural_max = 1330
    CHAPTER_NAME=0
    SECTION_NAME=1
    VERSE=2
    TRANSLATION=3
    EXPLANATION=4
    CHAPTER_INDEX=5
    SECTION_INDEX=6
    VERSE_INDEX=7
    KURAL_INDEX=8
    ERROR_CHAPTER_MSG = 'அதிகார எண் 133 க்குள் இருக்க வேண்டும்.'
    ERROR_VERSE_MSG = 'அதிகார குறள் வரிசை எண் 10 க்குள் இருக்க வேண்டும்.'
    ERROR_KURAL_MSG = 'குறள்  எண் 1330 க்குள் இருக்க வேண்டும்.'
    RANDOM_KURAL_MSG = 'சீரற்ற தேர்வு  (random choice):\n' 
    def __init__(self):
        """
            Column-0: Chapter Name, Col-1: Section Name, Col-2: Verse, Col-3: Translation, Col-4: Explanation
            Col-5: Chapter Index, Col-6: Section/Adhikaaram Index, Col-7: Verse Index, Col-8: Kural Index 
        """
        df=pd.read_csv('./Thirukural_With_Explanation2.csv',header=None,encoding='utf-8')
        self.df = df
    def _format_output(self, kural_id_list, random_kural=False):
        result =[]
        df = self.df
        for kural_id in kural_id_list:
            pd_series = df.loc[ (df[Thirukural.KURAL_INDEX]==kural_id)]
            chapter = ' '.join(pd_series[Thirukural.CHAPTER_NAME].values)
            adhikaram = ' '.join((pd_series[Thirukural.SECTION_NAME].values) + 
                                 str(pd_series[Thirukural.SECTION_INDEX].values)+" Kural:"+
                                 str(pd_series[Thirukural.KURAL_INDEX].values))
            verse_series = pd_series[Thirukural.VERSE].values
            verse = ' '.join(verse_series)
            verse1 = verse.replace('\t\t\t','\n').replace('\t',' ')
            meaning = ' '.join((pd_series[Thirukural.EXPLANATION].values))
            result.append(chapter+"\t"+adhikaram+"\n"+verse1+"\n"+meaning+"\n")
        random_kural_msg = ""
        if (random_kural):
            random_kural_msg = Thirukural.RANDOM_KURAL_MSG
        return random_kural_msg+'\n'.join(result)
    def get_statistics(self):
        df = self.df
        dfv = df[Thirukural.VERSE].str.translate(str.maketrans('', '', string.punctuation)).replace('\t',' ')
        kural_words = flatten_list([item.split() for item in dfv])
        print('Number of words in thirukuraL',len(kural_words))
        freq_words = frequency_of_occurrence(kural_words)
        print('Number of unique words in thirukuraL',len(freq_words))
        print('Top 10 words\n',freq_words[:10])        
    def contains(self, word):
        df = self.df
        temp_str = df.loc[df[Thirukural.VERSE].str.contains(word)][Thirukural.KURAL_INDEX]
        return self._format_output(temp_str)
    def endswith(self, word):
        if not word.endswith(END_OF_KURAL):
            word += END_OF_KURAL
        df = self.df
        end_char = '\t'
        if not word.endswith(end_char):
            word += end_char
        temp_str = df[df[Thirukural.VERSE].str.endswith(word)][Thirukural.KURAL_INDEX]
        return self._format_output(temp_str)
    def startswith(self, word):
        df = self.df
        end_char = '\t'
        if not word.endswith(end_char):
            word += end_char
        temp_str = df[df[Thirukural.VERSE].str.startswith(word)][Thirukural.KURAL_INDEX]
        return self._format_output(temp_str)
    def get(self, chapter_number=None,verse_number=None, kural_number=None):
        df = self.df
        temp_str = []
        random_kural = False
        if chapter_number is not None:
            if chapter_number > Thirukural.chapter_max:
                return Thirukural.ERROR_CHAPTER_MSG
            if verse_number is not None:
                if verse_number > Thirukural.verse_max:
                    return Thirukural.ERROR_VERSE_MSG
                temp_str = df.loc[ (df[Thirukural.SECTION_INDEX]==chapter_number) & (df[Thirukural.VERSE_INDEX]==verse_number)][Thirukural.KURAL_INDEX]
            else:
                temp_str = df.loc[df[Thirukural.SECTION_INDEX]==chapter_number][Thirukural.KURAL_INDEX]
        else:
            if kural_number is not None:
                if kural_number > Thirukural.kural_max:
                    return Thirukural.ERROR_KURAL_MSG
            else:
                kural_number = random.randint(1, Thirukural.kural_max)
                random_kural = True
            temp_str = df.loc[ (df[Thirukural.KURAL_INDEX]==kural_number) ][Thirukural.KURAL_INDEX]
        return self._format_output(temp_str,random_kural=random_kural)        
    def random(self):
        df = self.df
        kural_number = random.randint(1,Thirukural.kural_max)
        temp_str = df.loc[ (df[Thirukural.KURAL_INDEX]==kural_number)  ][Thirukural.KURAL_INDEX]
        return self._format_output(temp_str)
    
            
if __name__ == "__main__":
    """ Main script """
