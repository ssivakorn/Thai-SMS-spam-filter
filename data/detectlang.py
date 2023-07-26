# https://stackoverflow.com/questions/39142778/python-how-to-determine-the-language
# mixed_text=u'สวัสดีค่ะ วันนี้จะลองทดสอบของความ มีทั้งภาษาไทย และ English'

import sys
import fasttext

textfile = sys.argv[1]

model = fasttext.load_model('lid.176.ftz')

with open(textfile, 'r') as fp:
    for text in fp:
        text = text.strip()

        langs, cons = model.predict(text, k=1)  # top 3 matching languages

        # if '__label__th' in langs:
        #     print (f'TH|{text}')
        # else:
        #     print (f'Others|{text}')

        if '__label__th' in langs:
            print (text)
