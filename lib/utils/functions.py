import pandas as pd
import numpy as np
import os
import spacy
from spacy import displacy
import networkx as nx


import matplotlib.pyplot as plt


def ner(file_name):
    """
    Function to process text from a text file (.txt) using Spacy.
    params:
    file_name : a txt file as string

    returns:
    a processed doc file using Spacy English language model
    """
    nlp = spacy.load("en_core_web_sm")
    book_text = open(file_name).read()
    book_doc = nlp(book_text)

    return book_doc

def get_ne_list_per_sentence(spacy_doc):
    """
    Get a list of entities per sentence of a Spacy document and store in a dataframe

    params:
    spacy_doc : a Spacy processed document
    
    returns: 
    a dataframe containing the sentences and corresponding list pf recognized named entities 
    returns:
    """
    sent_entity_df = []

    #Loop through sentences, store named entity for each sentence
    for sent in spacy_doc.sents:
        entity_list = [ent.text for ent in sent.ents]
        sent_entity_df.append({"sentence": sent, "entities":entity_list})

    sent_entity_df = pd.DataFrame(sent_entity_df)

    return sent_entity_df

def filter_entity(ent_list, character_df):
    """
    Function to filter out non-character entities

    params: 
    ent_list : list of entities to be filtered
    character_df : a dataframe containing character names and character firstnames

    returns: 
    a list of entities that are characters (matching by names or firstnames)
    """
    
    return [ent for ent in ent_list
           if ent in list(character_df.character)
           or ent in list(character_df.character_firstname)]

def create_relationships(df,window_size):
    """
    Create dataframe of relationships based on the df dataframe and window size of n sentences.

    params:
    df : a dataframe containing a column called character_entities with the no of characters for each sentence of a document
    window_size : size of the window(for creating relationships between two adjacent characters in the text

    returns:
    a relationship dataframe containing 3 columns - source, target, value 
    """
    relationships = []

    for i in range(df.index[-1]):
        end_i = min(i + window_size, df.index[-1])
        char_list = sum((df.loc[i: end_i].character_entities), [])

        # Remove duplicated characters that are next to each other
        char_unique = [char_list[i] for i in range(len(char_list)) 
                       if (i==0) or char_list[i] != char_list[i-1]]

        if len(char_unique) > 1:
            for idx, a in enumerate(char_unique[:-1]):
                b = char_unique[idx + 1]
                relationships.append({"source": a, "target": b})
           
    relationship_df = pd.DataFrame(relationships)
    # Sort the cases with a->b and b->a
    relationship_df = pd.DataFrame(np.sort(relationship_df.values, axis = 1), 
                                   columns = relationship_df.columns)
    
    relationship_df.replace('Cirilla', 'Ciri', inplace=True)
    relationship_df["value"] = 1
    relationship_df = relationship_df.groupby(["source","target"], 
                                              sort=False, 
                                              as_index=False).sum()
                
    return relationship_df