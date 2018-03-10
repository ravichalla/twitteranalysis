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
 

#from dateutil import parser
#book = xlwt.Workbook(encoding="utf-8")
#sheet1 = book.add_sheet("Sheet1")

#def __init__(self):
 
 
def avg_retweet_count(input,leng):
    retweet_array=[]
    avg=0.0
    retweet_count_sum=0.0
    for x in range(0,leng):
        retweet_count= parsed_json[x]['retweet_count']
        retweet_array.append( retweet_count)
        retweet_count_sum+=retweet_array[x]
        retweet_avg=(retweet_count_sum/leng)
    return (retweet_avg)

def avg_favorite_count(input,leng):
    favorite_count_array=[]
    avg_favorite_count=0.0
    favorite_count_sum=0.0
    for x in range(0,leng):
        favorite_count_array.append( parsed_json[x]['favorite_count'])
        favorite_count_sum+=favorite_count_array[x]
        avg_favorite_count=favorite_count_sum/leng
    return (avg_favorite_count)

# sept 26 2017 -test period for 280 char incr ; Nov 7 2017 -rolled out 280 char tweet text
def avg_hashtag_count(input,leng):
    hashtag_array=[]
    hashtag_count_sum=0.0
    for x in range(0,leng):        
        #print(parsed_json[x]['favorite_count'])
        #print(parsed_json[x]['text'])
        #print(len( parsed_json[x]['entities']['hashtags'] ))
        hashtag_array.append(len(parsed_json[x]['entities']['hashtags'] ) )
        hashtag_count_sum+=hashtag_array[x]
    return (hashtag_count_sum/leng)

def avg_length_of_tweets(input,leng):
    old_tweet=0.0 
    new_tweet=0.0
    len_Oldtweet_sum=0.0
    len_newtweet_sum=0.0
    for x in range(0,leng):    
        tweet_text=parsed_json[x]['text']            
        ts = time.strptime(parsed_json[x]['created_at'],'%a %b %d %H:%M:%S +0000 %Y')
        dt = da_ti.fromtimestamp(mktime(ts))   
        if(dt < datetime.datetime(2017,9,27)):
            old_tweet=+1
            len_Oldtweet_sum=+len(parsed_json[x]['text'])
        else:
            new_tweet=+1
            len_newtweet_sum=+len(parsed_json[x]['text'])  
    old_tweet=1 if old_tweet<=0 else old_tweet
    new_tweet=1 if new_tweet<=0 else new_tweet
    return (len_Oldtweet_sum/old_tweet),(len_newtweet_sum/new_tweet)


def avg_url_count(input,leng):
    sum_urls_len_count=0.0
    urls_count_array=[]
    for x in range(0,leng):                
        urls_count=len(parsed_json[x]['entities']['urls'])#[0]#['url']
        #print(urls_count)
        urls_count_array.append(urls_count)
        sum_urls_len_count+=urls_count
    avg_url_count=(sum_urls_len_count/leng)
    return avg_url_count
    
def avg_emojis(input,leng):
    for x in range(0,leng):                
        #emoji's ( grimming face ex.)    
        tweet_text=parsed_json[x]['text']
        #tweet_text='\U1F601hj'
        #pattern = re.finditer(r'[\U0001F601-\U0001F64F]', tweet_text)
        pattern=r'[\U1F601-\U1F64F]'
        regex = re.compile(pattern, re.IGNORECASE)
        for match in regex.finditer(tweet_text):
            print "%s" % (match.group())
        print(regex)
        #todo
        #emoji_pattern = re.compile('[\U0001F300-\U0001F64F]')
        #emojis = emoji_pattern.findall(text)       
        #print(emojis)
        #todo

def avg_mentions_by_user(input,leng,num_elements):
    user_mention_name=[]
    most_mentioned=[]
    username_mentioned_sum=0.0
    i=0
    for x in range(0,leng):                
        username_mentioned_count=len(parsed_json[x]['entities']['user_mentions'])#[0]['name']
        username_mentioned_sum+=username_mentioned_count
        for y in range(0,username_mentioned_count):
            username_mentioned=parsed_json[x]['entities']['user_mentions'][y]['name']
            user_mention_name.append(username_mentioned)       
        cnt = Counter(user_mention_name)
        most_ment_counter=cnt.most_common(num_elements)
        username_mentioned_avg=(username_mentioned_sum/leng)  
        #print( type(cnt) )  --collections.counter type
    for name in most_ment_counter:
        most_mentioned.append(name)
    #widely_used_names=set([x for x in user_mention_name if user_mention_name.count(x) > 1]) #'set' data structure
    return username_mentioned_avg,most_mentioned#,top_most_name_list -todo

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
    #print(most_repeated_gram_counter)
        #username_mentioned_avg=(username_mentioned_sum/leng)  
        #print( type(cnt) )  --collections.counter type
    for gram in most_repeated_gram_counter:
        most_mentioned_grams.append(gram)
        #print(most_mentioned_grams)
    return str(output),(most_mentioned_grams)

def convertText(text, action):
    """
    Convert a string with embedded unicode characters to have XML entities instead
    - text, the text to convert
    - action, what to do with the unicode
    If it works return a string with the characters as XML entities
    If it fails return raise the exception
    """
    try:
        temp = unicode(text, "utf-8")
        fixed = unicodedata.normalize('NFKD', temp).encode('ASCII', action)
        return fixed
    except Exception, errorInfo:
        print errorInfo
        print "Unable to convert the Unicode characters to xml character entities"
        raise errorInfo

start_time = time.clock()

if __name__ == '__main__':
    directory = os.path.dirname(__file__)
    dir_to_take_files_name = os.path.join(directory, './Files_IO/INPUTfiles')
    
    influential_source_excel_filename=os.path.join(directory, './Files_IO/Influential_People.xlsx')
    get_influ_data=pd.read_excel(influential_source_excel_filename, sheet_name=0)
    
    column_area_from_infl_people = get_influ_data.ix[:,'Area']
    column_area_from_infl_people_list = column_area_from_infl_people.tolist()
    writer=pd.ExcelWriter('Tweets_allPersons_results.xlsx')
    fl_name,avg_retweet_all,avg_favorite_all,avg_url_count_all,avg_mention_count=[],[],[],[],[]
    avg_mention_all,avg_old_twe_len,avg_new_len=[],[],[]
    most_mentioned_names,Gram_1_array,Gram_2_array,Gram_3_array=[],[],[],[]
    tweet_length_array,person_area_name_array=[],[]
    Gram_1_array_all,most_mention_one_grams_array,most_mention_two_grams_array=[],[],[]
    most_mention_three_grams_array,Gram_2_array_all,Gram_3_array_all=[],[],[]
    
    punctuation = list(string.punctuation)
    
    for filename in os.listdir(dir_to_take_files_name):
        if filename.endswith(".json"):             
            complete_filename = os.path.join(dir_to_take_files_name,filename)
            name=str("https://twitter.com/")+filename[:-12]
            Area_column_object=get_influ_data.loc[get_influ_data['Name']== name , 'Area']
            Area=Area_column_object.iloc[0]
            person_area_name_array.append(Area)
            filejson=open(complete_filename,"r")
            text=filejson.read()    #string
            #raw = f.read()
            #print(type(filejson))
            #text=convertText(textREAD,unistr)
            
            parsed_json=simplejson.loads(text)   #dict
            leng=len(parsed_json)
            tweet_length_array.append(leng)
            filejson.close()
            len_Oldtweet_sum=0 
            len_newtweet_sum=0
 
            tokens = nltk.word_tokenize(text)

            fl_name.append(filename[:-12])
            
            avg_retweet_all.append( avg_retweet_count(parsed_json,leng)  )
            avg_favorite_all.append(avg_favorite_count(parsed_json,leng) )
            avg_url_count_all.append( avg_url_count(parsed_json,leng)  )            
            #avg_mention_count.append(avg_mentions_by_user(parsed_json,leng))
            #[ (k,Series(v)) for k,v in d.items() ]
            a,b=avg_mentions_by_user(parsed_json,leng,3) # number of top most elements,here 3      
            avg_mention_all.append(a)
            most_mentioned_names.append(str(b))
            x,y=avg_length_of_tweets(parsed_json,leng)
            avg_old_twe_len.append(int(x))
            avg_new_len.append(int(y))
            #avg_emojis(parsed_json,leng)        
                          
            #N-grams:
            tweet_text=parsed_json[x]['text']
            
            stop = stopwords.words('english') + punctuation + ['rt','RT','via']
            terms_stop = [term for term in preprocess(tweet_text) if term not in stop]
            print(terms_stop)
            
            #Gram_1_array_all.append(terms_stop)                   
            #bi grams
            tokens = nltk.word_tokenize(text)
            bgs = nltk.bigrams(tokens)
            print(tokens)
            print(bgs)
            #compute frequency distribution for all the bigrams in the text
            '''
            fdist = nltk.FreqDist(bgs)
            for k,v in fdist.items():
                print k,v
            
            one_grams,most_mention_one_grams=ngrams(parsed_json,1)
            Gram_1_array_all.append(str(one_grams) )
            most_mention_one_grams_array.append(str(most_mention_one_grams))
            '''
            
            
            two_grams,most_mention_two_grams=ngrams(parsed_json,2)
            Gram_2_array_all.append(two_grams)
            most_mention_two_grams_array.append(most_mention_two_grams)
                        
            three_grams,most_mention_three_grams=ngrams(parsed_json,3)
            Gram_3_array_all.append(three_grams)
            most_mention_three_grams_array.append(most_mention_three_grams)
         
         
            
            
            #print(b)      
    df=pd.DataFrame({'Names' : fl_name,'AREA':person_area_name_array,'retweet_avg' :avg_retweet_all,
                     'avg_favorite_all':avg_favorite_all,'avg_url_count':avg_url_count_all,
                     'avg_mention_count':avg_mention_all,
                     'most_mentioned_3':most_mentioned_names,
                     'Total_Num_Tweets':tweet_length_array ,
                     'avg_old_twe_len': avg_old_twe_len,
                     'avg_new_len':avg_new_len,                     
                     '1_gram':Gram_1_array_all,'2_gram':Gram_2_array_all,
                     '3_gram':Gram_3_array_all,
                      #'most_common_1_gram': most_mention_one_grams_array,
                      #'most_common_2_gram': most_mention_two_grams_array,
                      #'most_common_3_gram': most_mention_three_grams_array
                      } )
    '''d = dict( A = np.array([1,2])  ) 
    df2 = pd.DataFrame.from_dict(d, orient='index')
    bigdata = df.append(df2, ignore_index=True) '''

            #df=pd.DataFrame(data=np.array([fl_name],[x]),columns=['Names','tweet_avg'])
    df=df[['Names','AREA','retweet_avg','avg_favorite_all','avg_url_count',
           'avg_mention_count','most_mentioned_3','Total_Num_Tweets','avg_old_twe_len',
           'avg_new_len','1_gram','2_gram','3_gram'
           #,'most_common_1_gram','most_common_2_gram','most_common_3_gram'
           ]]
    #to get colummns in same order /required
    df.to_excel(writer,sheet_name='Tweet_Analysis',index=False)
    writer.save()
    print("file saved")
    print time.clock() - start_time, "seconds"

    #writable_book.save(file_path + '.out' + os.path.splitext(file_path)[-1])
    #book.save("tweet_results.xls")

    '''
    #n-grams getting
    print(ngrams(parsed_json,1))   
    print(ngrams(parsed_json,2))
    print(ngrams(parsed_json,3))
    '''
