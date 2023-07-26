import requests
import random
import time

from bs4 import BeautifulSoup

kDelay   = 5 # seconds
kBaseURL = 'https://www.freereceivesms.com/en/th'

def get_headers():
    version = f'{random.randint(530, 540)}.{random.randint(0, 50)}'
    return {
        'user-agent':   f'Mozilla/5.0 AppleWebKit/{version} '\
                        f'(KHTML, like Gecko) Chrome/100.0.0000.0 '\
                        f'Safari/{version}',
        'accept-language': 'en-US;en;q=0,th;q=0.8',
    }

def get_sms(prefix, pagenum):
    url = f'{kBaseURL}/{prefix}_{pagenum}.html'
    res = requests.get(url, headers=get_headers())
    if res.status_code != 200:
        yield None

    html = res.text
    soup = BeautifulSoup(html, "html.parser")

    for elem in soup.find_all('div', 'col-lg-8'):
        yield elem.get_text().replace('\n', ' ').replace('\r', ' ')

def main():
    # prefix = 8347
    # prefix = 8348
    prefix = 8349
    for p in range(1, 1000):
        for sms in get_sms(prefix=prefix, pagenum=p):
            print (sms)
        time.sleep (random.randint(2, kDelay))

if __name__ == "__main__": 
    main()