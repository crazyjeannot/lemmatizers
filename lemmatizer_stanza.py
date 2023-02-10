import pandas as pd
import numpy as np
from tqdm import tqdm
from glob import glob
import pickle
import stanza
from os import path
from unicodedata import normalize

path_name = '../corpus_chapitres_all_txt/*.txt'
path_test = '../corpus_chapitres_test/*.txt'

STANZA_PIPE = stanza.Pipeline(lang='fr', processors='tokenize, lemma', use_gpu=True)

def clean_text(txt):
    txt_res = normalize("NFKD", txt.replace('\xa0', ' '))
    txt_res = txt_res.replace('\\xa0', '')
    return txt_res


def pipeline_stanza(doc):

    pos_ko = ["NUM", "X", "SYM", "PUNCT", "SPACE"]

    list_token, list_lemma, list_pos = [], [], []
    nb_sentences, nb_tokens = 0, 0

    with open(doc, encoding="utf8") as file:
        text = file.readlines()
        text_clean = clean_text(str(text).lower())
        docs = STANZA_PIPE(text_clean)
        nb_sentences += len(docs.sentences)

        for sent in docs.sentences:
            nb_tokens += len(sent.words)
            for word in sent.words:
                #si le token est bien un mot on récupère son texte, son lemme et son pos
                if word.xpos not in pos_ko:
                    list_token.append(word.text)
                    list_lemma.append(word.lemma)
                    list_pos.append(word.xpos)

    return list_token, list_lemma, list_pos, nb_sentences, nb_tokens


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

        list_token, list_lemma, list_pos, nb_sentences, nb_tokens = pipeline_stanza(doc)#
        #list_token, list_lemma, nb_tokens = pipeline_spacy_fixed(doc)

        #print("PIPELINE SPACY ----------- OK")

        #print("NOMBRE TOKENS = ", nb_tokens)
        #print("NOMBRE SENTENCES = ", nb_sentences)

        nb_total_tokens += nb_tokens
        nb_total_sentences += nb_sentences

        main_list_token.append(list_token)
        main_list_lemma.append(list_lemma)
        main_list_pos.append(list_pos)
        main_list_index.append(doc_name)

        #print("\n\nEND PROCESSING NOVEL-----------")

    #print("\n\nEND PROCESSING CORPUS-----------")

    return main_list_lemma, main_list_token, main_list_pos, main_list_index


if __name__ == '__main__':
    main_list_lemma, main_list_token, main_list_pos, main_list_index = moulinette(path_test)
    with open('/data/jbarre/lemmatization/main_list_lemma_stanza.pkl', 'wb') as f1, open('/data/jbarre/lemmatization/main_list_token_stanza.pkl', 'wb') as f2, open('/data/jbarre/lemmatization/main_list_pos_stanza.pkl', 'wb') as f3, open('main_list_index_stanza.pkl', 'wb') as f4:
            pickle.dump(main_list_lemma, f1)
            pickle.dump(main_list_token, f2)
            pickle.dump(main_list_pos, f3)
            pickle.dump(main_list_index, f4)
