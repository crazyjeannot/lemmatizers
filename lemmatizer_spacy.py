import pandas as pd
import numpy as np
from tqdm import tqdm
from glob import glob
import pickle
from os import path
from unicodedata import normalize
import spacy

path_name = '../corpus_chapitres_all_txt/*.txt'
path_test = '../corpus_chapitres_test/*.txt'

spacy.prefer_gpu()
SPACY_PIPE = spacy.load('fr_core_news_lg')
#SPACY_PIPE = spacy.load('fr_core_news_sm')

SPACY_PIPE.max_length = 4000000
SPACY_PIPE.disable_pipes('ner')


def clean_text(txt):
    txt_res = normalize("NFKD", txt.replace('\xa0', ' '))
    txt_res = txt_res.replace('\\xa0', '')
    return txt_res

def pipeline_spacy_fixed(path):

    pos_ko = ["NUM", "X", "SYM", "PUNCT", "SPACE"]

    list_token, list_lemma, list_pos = [], [], []
    nombre_tokens = 0

    with open(path, encoding="utf8") as file:
        text = file.readlines()
        text_clean = clean_text(str(text).lower())
        ## split doc in 2 because of too long docs
        docs1 = SPACY_PIPE(text_clean[:len(text_clean)//2])
        docs2 = SPACY_PIPE(text_clean[len(text_clean)//2:])

        nombre_tokens = (len(docs1)+len(docs2))

        for token in docs1:
            #si le token est bien un mot on récupère son lemme
            if token.pos_ not in pos_ko:
                list_token.append(token.text)
                list_lemma.append(token.lemma_)
                #list_pos.append(token.pos_)
        for token in docs2:
            #si le token est bien un mot on récupère son lemme
            if token.pos_ not in pos_ko:
                list_token.append(token.text)
                list_lemma.append(token.lemma_)
                #list_pos.append(token.pos_)
    return list_token, list_lemma, nombre_tokens


def pipeline_spacy(path):

    pos_ko = ["NUM", "X", "SYM", "PUNCT", "SPACE"]

    list_token, list_lemma, list_pos = [], [], []
    nombre_tokens = 0

    with open(path, encoding="utf8") as file:
        text = file.readlines()
        text_clean = clean_text(str(text).lower())

        docs = SPACY_PIPE(text_clean)
        nombre_tokens += len(docs)

        for token in docs:
            #si le token est bien un mot on récupère son lemme
            if token.pos_ not in pos_ko:
                list_token.append(token.text)
                list_lemma.append(token.lemma_)
                list_pos.append(token.pos_)

    return list_token, list_lemma, list_pos, nombre_tokens

def moulinette(path_name):

    nb_total_tokens, nb_total_sentences = 0, 0
    main_list_token, main_list_lemma, main_list_pos, main_list_index = [], [], [], []

    #print("\n\nBEGIN PROCESSING CORPUS-----------")

    for doc in tqdm(glob(path_name)):

        #print("\n\nBEGIN PROCESSING NOVEL-----------")

        doc_name = path.splitext(path.basename(doc))[0]
        date = doc_name.split("_")[0]
        print(doc_name)

        #On recupere le texte des romans sous forme de listes de lemmes et de pos grâce à spacy

        #list_token, list_lemma, nb_sentences, nb_tokens = pipeline_stanza(doc)#list_pos,
        list_token, list_lemma, list_pos, nb_tokens = pipeline_spacy(doc)

        #print("PIPELINE SPACY ----------- OK")

        print("NOMBRE TOKENS = ", nb_tokens)
        #print("NOMBRE SENTENCES = ", nb_sentences)

        nb_total_tokens += nb_tokens
        #nb_total_sentences += nb_sentences

        main_list_token.append(list_token)
        main_list_lemma.append(list_lemma)
        main_list_pos.append(list_pos)
        main_list_index.append(doc_name)

        #print("\n\nEND PROCESSING NOVEL-----------")

    #print("\n\nEND PROCESSING CORPUS-----------")

    return main_list_lemma, main_list_token, main_list_pos, main_list_index #main_list_pos



if __name__ == '__main__':
    main_list_lemma, main_list_token, main_list_pos, main_list_index = moulinette(path_test)#path_test
    with open('/data/jbarre/lemmatization/main_list_lemma_spacy.pkl', 'wb') as f1, open('/data/jbarre/lemmatization/main_list_token_spacy.pkl', 'wb') as f2, open('/data/jbarre/lemmatization/main_list_pos_spacy.pkl', 'wb') as f3, open('/data/jbarre/lemmatization/main_list_index_spacy.pkl', 'wb') as f4:
            pickle.dump(main_list_lemma, f1)
            pickle.dump(main_list_token, f2)
            pickle.dump(main_list_pos, f3)
            pickle.dump(main_list_index, f4)
