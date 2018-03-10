import os,re,json,unicodedata
import simplejson,datetime,time
from datetime import datetime as da_ti
from datetime import date
from time import mktime
import numpy as np
from xlutils.copy import copy
import pandas as pd
from pandas import ExcelWriter
from pandas import ExcelFile
import xlsxwriter
from Finder.Type_Definitions import column
from collections import Counter
import math,collections
#from textbolb import TextBlob
from nltk.corpus import stopwords
import nltk,time,string
from nltk.tokenize import TweetTokenizer
 
emoticons_str = r"""
    (?:
        [:=;] # Eyes
        [oO\-]? # Nose (optional)
        [D\)\]\(\]/\\OpP] # Mouth
    )"""
 
regex_str = [
    emoticons_str,
    r'<[^>]+>', # HTML tags
    r'(?:@[\w_]+)', # @-mentions
    r"(?:\#+[\w_]+[\w\'_\-]*[\w_]+)", # hash-tags
    r'http[s]?://(?:[a-z]|[0-9]|[$-_@.&amp;+]|[!*\(\),]|(?:%[0-9a-f][0-9a-f]))+', # URLs
 
    r'(?:(?:\d+,?)+(?:\.?\d+)?)', # numbers
    r"(?:[a-z][a-z'\-_]+[a-z])", # words with - and '
    r'(?:[\w_]+)', # other words
    r'(?:\S)' # anything else
]
    
tokens_re = re.compile(r'('+'|'.join(regex_str)+')', re.VERBOSE | re.IGNORECASE)
emoticon_re = re.compile(r'^'+emoticons_str+'$', re.VERBOSE | re.IGNORECASE)
 
def tokenize(s):
    return tokens_re.findall(s)
 
def preprocess(s, lowercase=False):
    tokens = tokenize(s)
    if lowercase:
        tokens = [token if emoticon_re.search(token) else token.lower() for token in tokens]
    return tokens
 
#n grams
def ngrams(parsed_json, n):
    most_mentioned_grams,tweet_text_list = [],[] 
    output = {}
    for x in range(0,leng):  
        tweet_text=parsed_json[x]['text']
        print(tweet_text)
        #tweet_text.encode('utf-8')
        tweet_text.encode('ascii', 'ignore')
        print(tweet_text)
        tweet_text_list=tweet_text.split(' ') 
        for i in range(len(tweet_text_list)-n+1):
            g = ' '.join((tweet_text_list[i:i+n]))
            output.setdefault(g, 0)
            output[g] += 1
        #return output        
        cnt=Counter(output)
    most_repeated_gram_counter=cnt.most_common(20)
    for gram in most_repeated_gram_counter:
        most_mentioned_grams.append(gram)
        #print(most_mentioned_grams)
    return str(output),(most_mentioned_grams)

start_time = time.clock()

if __name__ == '__main__':
    directory = os.path.dirname(__file__)
    dir_to_take_files_name = os.path.join(directory, './Files_IO/INPUTfiles')   
    influential_source_excel_filename=os.path.join(directory, './Files_IO/Influential_People.xlsx')
    get_influ_data=pd.read_excel(influential_source_excel_filename, sheet_name=0)    
    column_area_from_infl_people = get_influ_data.ix[:,'Area']
    column_area_from_infl_people_list = column_area_from_infl_people.tolist()
    writer=pd.ExcelWriter('raviTweets_allPersons_results.xlsx')
    fl_name,avg_retweet_all,avg_favorite_all,avg_url_count_all,avg_mention_count=[],[],[],[],[]
    avg_mention_all,avg_old_twe_len,avg_new_len=[],[],[]
    most_mentioned_names,Gram_1_array,Gram_2_array,Gram_3_array=[],[],[],[]
    tweet_length_array,person_area_name_array=[],[]
    Gram_1_array_all,most_mention_one_grams_array,most_mention_two_grams_array=[],[],[]
    most_mention_three_grams_array,Gram_2_array_all,Gram_3_array_all=[],[],[]
    
    punctuation = list(string.punctuation)
    all_tweets=[]
    stop = stopwords.words('english') + punctuation + ['rt','RT','via']
    
    for filename in os.listdir(dir_to_take_files_name):
        if filename.endswith(".json"):             
            complete_filename = os.path.join(dir_to_take_files_name,filename)
            name=str("https://twitter.com/")+filename[:-12]
            Area_column_object=get_influ_data.loc[get_influ_data['Name']== name , 'Area']
            Area=Area_column_object.iloc[0]
            person_area_name_array.append(Area)
            
            with open(complete_filename, 'r') as f:
                raw = f.read()
            #raw_processed=unicode(raw,errors="ignore")
            raw_processed=str(raw)
            parsed_json=simplejson.loads(raw_processed)
            #print(parsed_json)
                    #terms_all = [term for term in preprocess(tweet['text'])]
            '''
            #filejson=open(complete_filename,"r")
            #text=filejson.read()    #string
            #raw = f.read()
            '''
            leng=len(parsed_json)
            all_tweets_string_format=' '
            for x in range(0,leng): 
                tweet_text=parsed_json[x]['text']
                #tweet_text=tweet_text.encode("utf-8")
                #print(tweet_text)
                #tweet_text=unicode(tweet_text,errors="ignore")
                #tweet_text=tweet_text.decode('cp1252')
                all_tweets_string_format +=tweet_text
                #all_tweets.append(tweet_text)
            #print(type(filejson))
            #text=convertText(textREAD,unistr)
            
            #parsed_json=simplejson.loads(text)   #dict
            #leng=len(parsed_json)
            #tweet_length_array.append(leng)
            #filejson.close()
            len_Oldtweet_sum=0 
            len_newtweet_sum=0
 
            #tokens = nltk.word_tokenize(text)            
            #avg_emojis(parsed_json,leng)        
                          
            #N-grams:
            terms_allowed = [term for term in preprocess(all_tweets_string_format) if term not in stop]
            
            #print(type(terms_allowed))
            #print(terms_allowed)
     
           
    df=pd.DataFrame({ 'Names' : fl_name,'AREA':person_area_name_array                 
                     #'1_gram':terms_allowed
                     #'2_gram':Gram_2_array_all,'3_gram':Gram_3_array_all,
                      #'most_common_1_gram': most_mention_one_grams_array,
                      #'most_common_2_gram': most_mention_two_grams_array,
                      #'most_common_3_gram': most_mention_three_grams_array
                      } )        
         
    # df=df[['Names','AREA','1_gram'
           #,'most_common_1_gram','most_common_2_gram','most_common_3_gram'
    #      ]]
    '''
    a = {'Names' : fl_name,'AREA':person_area_name_array,                 
                     '1_gram':terms_allowed}
    df = pd.DataFrame.from_dict(a, orient='index')
    '''
    
    #df.transpose()
    df.to_excel(writer,sheet_name='partial_Tweet_Analysis',index=False)
    
    print("file saved ")
    print time.clock() - start_time, "seconds"

    #writable_book.save(file_path + '.out' + os.path.splitext(file_path)[-1])
    #book.save("tweet_results.xls")

    '''
    #n-grams getting
    print(ngrams(parsed_json,1))   
    print(ngrams(parsed_json,2))
    print(ngrams(parsed_json,3))
    '''
