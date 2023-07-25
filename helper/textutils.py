#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import logging
import numpy as np
import re
import random
import pythainlp

print (pythainlp.__version__)

from pythainlp import word_tokenize, Tokenizer
from pythainlp.corpus.common import thai_words
from pythainlp.util import dict_trie

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

logger = logging.getLogger(__name__)


# In[ ]:


# Format text
# Spaces
# Special characters -> space character
# Add 1 space character between different languages
# Join multiple lines with space character
# Multiple spaces -> 1 space character

# Others
# Link/URL -> <link>
# continue digits (129373) -> <digits>
# Emojis -> <emoji>


# In[9]:


# kLinkPattern =  r'\b((?:https?://)?(?:(?:www\.)?(?:[\da-z\.-]+)\.(?:[a-z]{2,6})|(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)|(?:(?:[0-9a-fA-F]{1,4}:){7,7}[0-9a-fA-F]{1,4}|(?:[0-9a-fA-F]{1,4}:){1,7}:|(?:[0-9a-fA-F]{1,4}:){1,6}:[0-9a-fA-F]{1,4}|(?:[0-9a-fA-F]{1,4}:){1,5}(?::[0-9a-fA-F]{1,4}){1,2}|(?:[0-9a-fA-F]{1,4}:){1,4}(?::[0-9a-fA-F]{1,4}){1,3}|(?:[0-9a-fA-F]{1,4}:){1,3}(?::[0-9a-fA-F]{1,4}){1,4}|(?:[0-9a-fA-F]{1,4}:){1,2}(?::[0-9a-fA-F]{1,4}){1,5}|[0-9a-fA-F]{1,4}:(?:(?::[0-9a-fA-F]{1,4}){1,6})|:(?:(?::[0-9a-fA-F]{1,4}){1,7}|:)|fe80:(?::[0-9a-fA-F]{0,4}){0,4}%[0-9a-zA-Z]{1,}|::(?:ffff(?::0{1,4}){0,1}:){0,1}(?:(?:25[0-5]|(?:2[0-4]|1{0,1}[0-9]){0,1}[0-9])\.){3,3}(?:25[0-5]|(?:2[0-4]|1{0,1}[0-9]){0,1}[0-9])|(?:[0-9a-fA-F]{1,4}:){1,4}:(?:(?:25[0-5]|(?:2[0-4]|1{0,1}[0-9]){0,1}[0-9])\.){3,3}(?:25[0-5]|(?:2[0-4]|1{0,1}[0-9]){0,1}[0-9])))(?::[0-9]{1,4}|[1-5][0-9]{4}|6[0-4][0-9]{3}|65[0-4][0-9]{2}|655[0-2][0-9]|6553[0-5])?(?:/[\w\.-\?\-.]*)*/?)\b'
kLinkPattern = r'\b((?:https?://)?(?:(?:www\.)?(?:[\da-zA-Z\.-]+)\.(?:[a-zA-Z]{2,6})|(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)|(?:(?:[0-9a-fA-F]{1,4}:){7,7}[0-9a-fA-F]{1,4}|(?:[0-9a-fA-F]{1,4}:){1,7}:|(?:[0-9a-fA-F]{1,4}:){1,6}:[0-9a-fA-F]{1,4}|(?:[0-9a-fA-F]{1,4}:){1,5}(?::[0-9a-fA-F]{1,4}){1,2}|(?:[0-9a-fA-F]{1,4}:){1,4}(?::[0-9a-fA-F]{1,4}){1,3}|(?:[0-9a-fA-F]{1,4}:){1,3}(?::[0-9a-fA-F]{1,4}){1,4}|(?:[0-9a-fA-F]{1,4}:){1,2}(?::[0-9a-fA-F]{1,4}){1,5}|[0-9a-fA-F]{1,4}:(?:(?::[0-9a-fA-F]{1,4}){1,6})|:(?:(?::[0-9a-fA-F]{1,4}){1,7}|:)|fe80:(?::[0-9a-fA-F]{0,4}){0,4}%[0-9a-zA-Z]{1,}|::(?:ffff(?::0{1,4}){0,1}:){0,1}(?:(?:25[0-5]|(?:2[0-4]|1{0,1}[0-9]){0,1}[0-9])\.){3,3}(?:25[0-5]|(?:2[0-4]|1{0,1}[0-9]){0,1}[0-9])|(?:[0-9a-fA-F]{1,4}:){1,4}:(?:(?:25[0-5]|(?:2[0-4]|1{0,1}[0-9]){0,1}[0-9])\.){3,3}(?:25[0-5]|(?:2[0-4]|1{0,1}[0-9]){0,1}[0-9])))(?::[0-9]{1,4}|[1-5][0-9]{4}|6[0-4][0-9]{3}|65[0-4][0-9]{2}|655[0-2][0-9]|6553[0-5])?(?:/[^ ]+)*/?)\b'
kOTPPattern  = r'(^|[^\d\.\,\/])+(\d{4,6})($|[^\d\.\,\/])+'

kLineHosts = [
    'lin.ee/',
    'line.me/',
    'line://',
    'liff.line.me/',
]

kShortLinkHosts = [
    'bit.ly/',
    'cutt.ly/',
    'tinyurl.com/',
    'tiny.cc/',
    'rb.gy/',
    'ow.ly/',
    't.co/',
]

def add_space_before_http(text):
    text = text.replace('http://', ' http://')
    text = text.replace('https://', ' https://')
    return text
    
def get_all_links(text):
    return re.findall(kLinkPattern, text)  

def _does_match_hostname(hostnames, link):
    link = link.lower().strip()
    for host in hostnames:
        if (link.startswith(host) or 
            link.startswith(f'http://{host}') or \
            link.startswith(f'https://{host}')):
            return True

    return False
    

def has_short_link(links):
    # List of public popular short link services (no register required)
    
    for lnk in links:
        if _does_match_hostname(kShortLinkHosts, lnk):
            return True
    return False
              

def has_lineapp(links):
    for lnk in links:
        if _does_match_hostname(kLineHosts, lnk):
            return True
    return False


def get_all_OTP(text):
    OTPs = list()
    for match in re.findall(kOTPPattern, text):
        OTPs.append(match[1])
    
    return OTPs

    
def remove_all_links(links, text):
    
    for lnk in links:
        if _does_match_hostname(kLineHosts, lnk):
            text = text.replace(lnk, '<LINE>')
        elif _does_match_hostname(kShortLinkHosts, lnk):
            text = text.replace(lnk, '<SHORT-LINK>')
    
    return re.sub(kLinkPattern, ' <LINK> ', text)

def remove_all_otp(text, otp_lst):
    for otp in otp_lst:
        text = text.replace(otp, ' <OTP> ')
    return text


kBlankPattern = re.compile(r'\s+')
def remove_blanks(text):
    return kBlankPattern.sub(' ', text).strip()


def sanitize_text(text, remove_links=True):
    """Sanitizing text"""
    
    text = add_space_before_http(text)
    
    if remove_links:
        links = get_all_links(text) 
        text  = remove_all_links(links, text)
    
    text = remove_blanks(text)
    text = text.replace('เเ', 'แ') # Fix Thai Grammar
    return text


# In[ ]:





# In[ ]:


# Tokenize
# Build dictionary (word list)

kSpecialChars = '''!"#$&%()*+,-./:;<=>?@[\]^_\'{|}~ '''
kSpecialChars = set(kSpecialChars)

def is_special_char(word):
    if word in kSpecialChars:
        return True
    return False

def split_special_chars(words):
    new_words = list()
    for word in words:
        if word == ''.join(list(filter(lambda c: is_special_char(c), word))):
            # should split special characters
            new_words += word
        else:
            new_words.append(word)

    return new_words

kOmittedWords = {'<OTP>'}
kCustomWords = {'<LINK>',
                '<LINE>',
                '<SHORT-LINK>',
                '<NAME>'}

kOtherWords = set()
current_dir = os.path.abspath(os.path.dirname(__file__))
custom_words_file = os.path.join(current_dir, "custom_words.txt")
with open(custom_words_file) as fp:
    for line in fp:
        kOtherWords.add(line.strip())
        
# kOtherWords  = {
#     'ม.ค.', 'ก.พ.', 'มี.ค.', 'เม.ย.', 'พ.ค.', 'มิ.ย.', 'ก.ค.', 'ส.ค.', 'ก.ย.', 'ต.ค.', 'พ.ย.', 'ธ.ค.',
#     'คริปโต', 'บิตคอยต์', 'เครดิด', 'เครดิต', 'วอเลต', 'วอลเล็ต', 'แอป', 'ๆ', '!', 'บ/ช',
#     'แอ็กเคานต์', 'แอปพลิเคชัน', 'แชต', 'เช็ก', 'โค้ด', 'คอมเมนต์', 'คอนเทนต์', 'ก๊อปปี้', 
#     'เคานต์ดาวน์', 'อีเวนต์', 'เกม', 'เกมส์', 'ไฮไลต์', 'แจ็กพอต', 'เมสเซ็นเจอร์', 'มิวสิก', 'เน็ตเวิร์ก' , 'โน้ตบุ๊ก',
#     'พาร์ตไทม์', 'พอยต์', 'ปรินต์', 'โพรไฟล์', 'สคริปต์', 'เซนเซอร์', 'เซ็กซ์' , 'ชอปปิง', 'ชอปปิ้ง', 'สมาร์ตโฟน', 'ซับสไครบ์',
#     'เต็นท์', 'อัปเดต' , 'อัพเดท', 'อัปเดท', 'อัพเดท', 'เวอร์ชัน', 'เวอร์ชั่น', 'วิดีโอ', 'วีดีโอ', 'วอลล์เปเปอร์', 'เวิร์กชอป', 
#     'เอกซ์', 'เอ็กซ์', 'เอ๊กซ์',
#     'ลิงก์', 'ยูส', 'ยี่กี', 'เทส', 'บาคาร่า', 'ไลน์', 'ลาซาด้า', 'โปรโมชั่น', 'โปร', 'บิต', 'อีสปอร์ต', '฿',
#     'พ.ร.บ', 'กธ.'
# }

kReplaceWords = {
    '฿': 'บาท',
    'บ': 'บาท',
    'น': 'นาฬิกา',
    
}

# Thai words
custom_words_list = set(thai_words())
custom_words_list.update(kCustomWords.union(kOtherWords))
trie = dict_trie(dict_source=custom_words_list)
custom_tokenizer = Tokenizer(custom_dict=trie, engine='newmm')

def tokenize(text, clean=False):

    # tokenize
    words = custom_tokenizer.word_tokenize(text)
    words = split_special_chars(words)

    words = list(filter(lambda w: not is_special_char(w), words))
    words = [kReplaceWords[w] if w in kReplaceWords else w for w in words]  

    if not clean:
        return words

    # tokenize + clean
    clean_words = list()
    for word in words:
        if word in kOmittedWords:
            continue
        if word in kCustomWords:
            clean_words.append(word)
            continue

        # kRegExp = r"[ก-๙a-zA-Z']+"
        kRegExp = r"[\u0E00-\u0E7Fa-zA-Z' ]+"
        word = ''.join(re.findall(kRegExp, word))
        
        if (word and
            word not in kOmittedWords and 
            word not in kCustomWords):
            word = re.sub(r'[A-Z]', lambda m: m.group(0).lower(), word)
        
        clean_words += [word] if word else []
        
    return clean_words


class Dictionary:

    def __init__(self):
        self.cvec  = None
        self.tfvec = None
        self.word_lists = list()
        self.bow      = None
        self.tfmatrix = None
        self.custom_vocab = list()

    def add(self, text, clean=True):
        words = tokenize(text, clean)
        self.word_lists.append(','.join(words))
        
        for w in words:
            self.custom_vocab.append(w)
        self.custom_vocab = list(set(self.custom_vocab))
        
    
    def add_completed(self):
        self.cvec = CountVectorizer(tokenizer=lambda x:x.split(','), vocabulary=self.custom_vocab)
        self.bow = self.cvec.fit_transform(self.word_lists)        

        self.tfvec = TfidfVectorizer(tokenizer=lambda x:x.split(','), vocabulary=self.custom_vocab)
        self.tfidf = self.tfvec.fit_transform(self.word_lists)
    
    def get_cvec(self):
        return self.cvec
    
    def get_tfvec(self):
        return self.tfvec
    
    def get_vocabulary(self, vec_type='cvec'):
        
        if vec_type == 'cvec':
            return self.cvec.vocabulary_
        elif vec_type == 'tfidf':
            return self.tfvec.vocabulary_
        else:
            raise RuntimeError('No vectorizer type found')
        
    def get_vocabulary_count(self, vec_type='cvec'):
        if vec_type == 'cvec':
            return len(self.cvec.vocabulary_)
        elif vec_type == 'tfidf':
            return len(self.tfvec.vocabulary_)
        else:
            raise RuntimeError('No vectorizer type found')  
    
    def get_bag_of_words(self):
        return self.bow.toarray()
    
    def get_tfidf(self):
        return self.tfidf.toarray()

        


# In[ ]:


def padded_seq(words, word_index, maxlen=50, ignore_err=False):
    
    sequence = list()
    for w in words:
        if w not in word_index:
            error_msg = f'Found word not in word_index: word={w}'
            if ignore_err:
                logger.warning(error_msg)
                continue
            else:
                raise RuntimeError(error_msg)
            
        sequence.append(word_index[w])
        
    if len(sequence) < maxlen:
        sequence += [0] * (maxlen - len(sequence))
    
    return sequence

def padded_seq_from_text(text, word_index, maxlen=50, ignore_err=False):
    
    words = tokenize(text, clean=True)
    return padded_seq(words, word_index, maxlen, ignore_err=ignore_err)


# In[1]:


def bow_array(words, word_index, ignore_err=False):
    bow_array = np.zeros(len(word_index))
    for w in words:
        if w not in word_index:
            error_msg = f'Found word not in word_index: word={w}'
            if ignore_err:
                logger.warning(error_msg)
                continue
                
        bow_array[word_index[w]] += 1
    return bow_array
    
    
def bow_array_from_text(text, word_index, ignore_err=False):
    words = tokenize(text, clean=True)
    return bow_array(words, word_index, ignore_err=ignore_err)
    


# In[ ]:


def tfidf_array(words, word_index, idf_values, ignore_err=False):
    tfidf_array = np.zeros(len(word_index))
    for w in words:
        if w not in word_index:
            error_msg = f'Found word not in word_index: word={w}'
            if ignore_err:
                logger.warning(error_msg)
                continue
                
        # tfidf_array[word_index[w]] += 1
        tf  = words.count(w) / len(words) # term frequency
        idf = idf_values[word_index[w]]   # inverse document frequency
        tfidf_array[word_index[w]] = tf * idf
        
    return tfidf_array
    
    
def tfidf_array_from_text(text, word_index, idf_values, ignore_err=False):
    words = tokenize(text, clean=True)
    return tfidf_array(words, word_index, idf_values, ignore_err=ignore_err)

