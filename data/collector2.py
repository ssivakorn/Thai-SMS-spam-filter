import requests
from requests_html import HTMLSession


kURL = 'https://sms24.me/en/number-66825410506'
# pagination https://sms24.me/en/number-66825410506/2/
kSEL = f'body > div.container-fluid.mt-3.p-0 > div.row.g-0 > div.col-xl-8 > div.card.bg-white div.card-body > div.container-fluid.p-0 > div.row'


def get_sms(url): 
    session = HTMLSession()
    r = session.get(url)
    for elem in r.html.find(kSEL, first=False):
        yield elem.text

def main():
    for pnum in range(1, 2):
        for text in get_sms(f'{kURL}'):
            print (text)


if __name__ == '__main__':
    main()