import sys

input_file = sys.argv[1]


kOptions = {
    '0': 'OK',
    '1': 'scam',
    '2': 'OTP',
    '3': 'spam',
    '4': 'unsure',
    '5': 'out of Scope (e.g., other languages)',
}

kOptionMsg = ''
for k in kOptions.keys():
    kOptionMsg += f'[{k}]: {kOptions[k]}'
kOptionMsg = f'select: {kOptions}: '

labelled_text = []

with open(input_file) as fp:
    for text in fp.readlines():
        print ('=' * 80)
        print (text)
        print ('=' * 80)

        selected = input(kOptionMsg)
        
        while not selected in kOptions:
            print (f'Incorrect option selected, try again!')
            selected = input(kOptionMsg)

        if selected == '5':
            continue
        
        labelled_text.append((kOptions[selected], text))

with open(f'{input_file}_labelled.txt', 'w', encoding='utf8') as fp:
    for label, text in labelled_text:
        fp.write(f'{label}|{text}')