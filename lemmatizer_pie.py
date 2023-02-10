import pandas as pd
import numpy as np
from glob import glob
import pickle
from os import path
from unicodedata import normalize
from tqdm import tqdm

from pie_extended.cli.utils import get_tagger
from pie_extended.models.dum.imports import get_iterator_and_processor

model_name = "fr"
tagger = get_tagger(model_name, batch_size=256, device="gpu", model_path=None)


path_name = '../corpus_chapitres_all_txt/*.txt'
path_test = '../corpus_chapitres_test/*.txt'



def clean_text(txt):
    txt_res = normalize("NFKD", txt.replace('\xa0', ' '))
    txt_res = txt_res.replace('\\xa0', '')
    return txt_res

def nettoie_lemma(list_lemma, list_pos):
    pos_ok = ["pre", "adv", "det", "conj", "pron", "adp", "art"]
    list_nettoie = []
    if len(list_lemma) == len(list_pos):
        for i in range(0, len(list_lemma)-1):
            if list_pos[i].split("(")[0] in pos_ok:
                list_nettoie.append(list_lemma[i])
    return list_nettoie


def pipeline_pie(doc):
    nombre_tokens = 0
    with open(doc, encoding="utf8") as file:
        lignes = file.readlines()

        for ligne in lignes:
            iterator, processor = get_iterator_and_processor()
            annotation = tagger.tag_str(ligne, iterator=iterator, processor=processor)

    list_token = list(annotation['form'])
    nombre_tokens = list_token

    list_lemma = nettoie_lemma(annotation['lemma'], annotation['pos'])

    return list_token, list_lemma, nombre_tokens


def moulinette(path_name):

    nb_total_tokens = 0
    main_list_token, main_list_lemma, main_list_index = [], [], []

    #print("\n\nBEGIN PROCESSING CORPUS-----------")

    for doc in tqdm(glob(path_name)):

        #print("\n\nBEGIN PROCESSING NOVEL-----------")

        doc_name = path.splitext(path.basename(doc))[0]
        date = doc_name.split("_")[0]
        print(doc_name)

        #On recupere le texte des romans sous forme de listes de lemmes et de pos grâce à spacy

        list_token, list_lemma, nb_tokens = pipeline_pie(doc)#
        #list_token, list_lemma, nb_tokens = pipeline_spacy_fixed(doc)

        #print("PIPELINE SPACY ----------- OK")

        #print("NOMBRE TOKENS = ", nb_tokens)
        #print("NOMBRE SENTENCES = ", nb_sentences)

        nb_total_tokens += nb_tokens
        main_list_token.append(list_token)
        main_list_lemma.append(list_lemma)
        main_list_index.append(doc_name)

        #print("\n\nEND PROCESSING NOVEL-----------")

    #print("\n\nEND PROCESSING CORPUS-----------")

    return main_list_lemma, main_list_token, main_list_index


if __name__ == '__main__':
    main_list_lemma, main_list_token, main_list_index = moulinette(path_test)
    with open('/data/jbarre/lemmatization/main_list_lemma_pie.pkl', 'wb') as f1, open('/data/jbarre/lemmatization/main_list_token_pie.pkl', 'wb') as f2, open('/data/jbarre/lemmatization/main_list_index_pie.pkl', 'wb') as f3:
            pickle.dump(main_list_lemma, f1)
            pickle.dump(main_list_token, f2)
            pickle.dump(main_list_index, f3)
