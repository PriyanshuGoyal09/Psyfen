# importing required libraries
import re
import pandas as pd
import pytesseract
from pdf2image import convert_from_path
from os import listdir
from os.path import isfile, join


def convert_pdf(file):
    pages = convert_from_path(file)
    text = ''
    for page in pages:
        txt = pytesseract.image_to_string(page, lang='eng')
        text += txt

    return text


def text_cleaner(raw_text):
    # removing \n,multiple dots,white spaces and underscores
    cleaned_text = raw_text.replace('\n', '')
    cleaned_text = cleaned_text.replace('\x0a', '')
    cleaned_text = cleaned_text.replace("\x0c", " ")

    regex = r"\ \.\ "
    subst = "."
    cleaned_text = re.sub(regex, subst, cleaned_text, 0)

    regex = "_"
    subst = " "
    cleaned_text = re.sub(regex, subst, cleaned_text, 0)  # Get rid of underscores

    regex = "--+"
    subst = " "
    cleaned_text = re.sub(regex, subst, cleaned_text, 0)

    regex = r"\*+"
    subst = "*"
    cleaned_text = re.sub(regex, subst, cleaned_text, 0)

    regex = r"\ +"
    subst = " "
    cleaned_text = re.sub(regex, subst, cleaned_text, 0)

    return cleaned_text


def data_organizer(clean_text):
    short_text = clean_text[:1000]
    len_text = len(short_text)
    return clean_text, short_text, len_text


def preprocess(file):
    raw_text = convert_pdf(file)
    cleaned_text = text_cleaner(raw_text)
    cleaned, short, len1 = data_organizer(cleaned_text)
    return cleaned, short, len1


def multi_file():
    clean_text_li = []
    short_text_li = []
    len_li = []
    dir_path = str(input('Enter dir path'))
    files = [f for f in listdir(dir_path) if isfile(join(dir_path, f))]
    for file in files:
        file_path = dir_path + '/' + file
        cleaned, short, len2 = preprocess(file_path)
        clean_text_li.append(cleaned)
        short_text_li.append(short)
        len_li.append(len)
    data_dict = {
        'Full_Text': clean_text_li,
        'Short_Text': short_text_li,
        'Length': len_li
    }
    exp_df = pd.DataFrame(data_dict)
    exp_df.to_csv('text.csv', index=False)
