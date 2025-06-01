import os
import re
import json
import shutil
import pandas as pd
import argparse
import sklearn
import time
import warnings
import numpy as np
import glob
from numpy.random import RandomState
from joblib import Parallel, delayed, cpu_count
from trankit import Pipeline
from collections import Counter


def read_data(path, index_col=None, sep=','):
    """
    Returns the data as pandas dataframe. Can only real csv and pickle format. 
    """
    if path.endswith('.csv') or path.endswith('.tsv'):
        df = pd.read_csv(glob.glob(path)[0], index_col=index_col, sep=sep)
        return df
    elif path.endswith('.pkl'):
        df = pd.read_pickle(glob.glob(path)[0])
        return df
    else:
        warnings.warn('Unknown file format. Please provide your data in .csv or .pkl format.')
    

def draw_random_sample(df, random_number, n):
    """
    Returns the dataframe with a randomly drawn subsample of the rows.
    """
    random_state = RandomState(random_number)
    df = df.sample(n=n, random_state=random_state)

    return df


def preprocess_transcripts(df, text_column):
    """
    Returns the dataframe with preprocessed text column.
    """
    for i in range(len(df[text_column])):
        if pd.isna(df.loc[i, text_column]):
            df.loc[i, text_column] = ''
            pass
        df.loc[i, text_column] = df.loc[i, text_column].lower()
        df.loc[i, text_column]= re.sub(r'[^\w\s]', '', df.loc[i, text_column])
        df.loc[i, text_column] = re.sub(r'\d+', '', df.loc[i, text_column])
        df.loc[i, text_column] = re.sub(r'\n+', ' ', df.loc[i, text_column])
        df.loc[i, text_column] = re.sub(r'\s+', ' ', df.loc[i, text_column])
        df.loc[i, text_column] = df.loc[i, text_column].rstrip()

    return df


def apply_trankit_pipeline(utterance):
    """
    Returns a list of strings containing the lemmatized words of an utterance,
    computed with the Trankit Finnish Pipeline.
    """
    if isinstance(utterance, str) and not len(utterance) == 0: 
        output = trankit_pipe(utterance, is_sent=True)

        return [re.sub(r'#', '', token.get('lemma', '')) for token in output['tokens']]
    else: 

        return ''


def parallel_lemmatizer(transcript_list, lemma_column, num_workers=5, batch_size=100):
    """
    Returns a list of strings containing the lemmatized words for every utterance in the 
    dataframe using the Trankit Finnish Pipeline. Parallel implementation. 
    """
    print(f'number of workers: {num_workers}')
    text_lemmatized = Parallel(n_jobs=num_workers, return_as='list', batch_size=batch_size)(
        delayed(apply_trankit_pipeline)(text_sample) for text_sample in transcript_list)
    df[lemma_column] = text_lemmatized

    return df


def get_self_scores(utterance):
    """
    Returns a dictionary mapping each lemma in an utterance to a sorted list 
    of corresponding sentiment categories from the SELF lexicon.
    """
    sentiment_cols = [
        'disgust', 'anger', 'fear', 'sadness', 'joy',
        'trust', 'anticipation', 'surprise', 'positive', 'negative'
    ]
    scores = dict()
    for lemma in utterance:
        if not isinstance(lemma, str):
            continue

        # look up matches for one lemma in SELF column named 'word'
        matches = self[self['word'] == lemma]
        if not matches.empty:
            # collect non-zero sentiment categories for all matching rows
            features = set()
            for _, row in matches.iterrows():
                features.update([col for col in sentiment_cols if row[col] != 0.0])
            if features:
                scores[lemma] = sorted(features) 

    return scores


def get_feil_scores(utterance):
    """
    Returns a dictionary mapping each lemma in an utterance to a sorted list 
    of corresponding sentiment categories from the FEIL lexicon.
    """
    scores = dict()
    for lemma in utterance:
        if not isinstance(lemma, str):
            continue

        # look up matches for one lemma in FEIL column named 'finnish-fi'
        matches = feil[feil['finnish-fi'] == lemma]
        if not matches.empty:
            # collect emotion categories in FEIL column named 'emotion'
            emotions = set(matches['emotion'])  
            if emotions:
                scores[lemma] = sorted(emotions) 
    
    return scores


def combine_scores(utterance):
    """
    Returns a dictionary mapping each lemma in an utterance to a sorted list 
    of corresponding sentiment categories from the SELF and FEIL lexica combined.
    """
    self_score = get_self_scores(utterance)
    feil_score = get_feil_scores(utterance)
    try:
        for k, v in feil_score.items(): 
            # since SELF has more labels, it is introduced first 
            self_score.setdefault(k, v) 
            for i in range(len(v)):
                if v[i] not in self_score[k]: 
                    self_score[k].append(v[i])
    except KeyError:
        pass
    return self_score 


def calculate_utterance_score(sentence):
    """
    Returns a dictionary mapping each sentiment category found in an utterance
    to a contonuous score. The score is the ratio of words falling into the respective
    sentiment category divided by the number of words in an utterance.
    """
    word_count = len(sentence)
    # get SELF and FEIL sentiment categories
    scores = combine_scores(sentence)
    sentiments = [sentiment for score in scores.values() for sentiment in score]
    counts = Counter(sentiments)
    result = dict()
    # calculate ratio
    for i in counts:
        sentiment_score = counts[i]/word_count
        result[i] = sentiment_score

    return result


def map_to_valence(utterance, continuous=False):
    """
    Returns either a discrete valence label (negative, positive, neutral) or 
    a dictionary containing the labels for one utterance and a corresponding 
    continuous score between 0 and 1. The score is the ratio of words with 
    a given label divided by the number of words in an utterance. 
    """
    combined_score = combine_scores(utterance)

    # in case no word in an utterance was found in the lexia, assume neutrality
    if len(combined_score) == 0:
        if continuous:
            return {'neutral': 0.9999}
        else:
           return 'neutral'
        
    # mapping from sentiment categories to valence label
    sentiment_to_valence = {
        'disgust': 0, 'fear': 0, 'anger': 0, 'sadness': 0, 'negative': 0,
        'trust': 1, 'joy': 1, 'positive': 1, 'surprise': 1, 'anticipation': 1
    }

    labels = ['negative', 'positive']
    valence = []
    for sentiments in combined_score.values():
        valence_values = [0, 0]
        # majority vote: count positive and negative connotations of one lemma
        for s in sentiments:
            idx = sentiment_to_valence.get(s, -1)
            if idx == -1:
                warnings.warn(f'Unknown sentiment category {s} in method map_to_valence_disc')
            else:
                valence_values[idx] += 1
        # small bias towards negative class
        valence.append(labels[0 if valence_values[0] >= valence_values[1] else 1])
    
    counts = Counter(valence)
    word_count = len(utterance)
    # calculate valence score
    result = {label: count / word_count for label, count in counts.items()}
    if continuous:
        return result
    else:
        return max(result, key=result.get)
    
    
def map_to_valence_disc_fast(utterance_score):
    """
    Returns a discrete valence label (negative, positive, neutral).
    Fast implementation: Takes the valence of the first maximum score 
    in the sorted sentiment score dictionary as valence label. 
    """
    if len(utterance_score) == 0:
        return 'neutral'
    else:
        # mapping from sentiment categories to valence label
        neg_keys = ['disgust', 'fear', 'anger', 'sadness', 'negative']
        pos_keys = ['trust', 'joy', 'positive', 'surprise', 'anticipation']
        # find maximum score
        sorted_scores = sorted(utterance_score, key=utterance_score.get, reverse=True)
        max_emotion = list(sorted_scores)[0]

        if max_emotion in neg_keys:
            return 'negative'
        elif max_emotion in pos_keys:
            return 'positive'
        else:
            warnings.warn(f'Unknown sentiment category {max_emotion} in method map_to_valence_disc_fast')


# main
if __name__=='__main__':

    # 0. Argument Parser
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str,help='config file')
    args = vars(parser.parse_args())

    with open(args['config']) as f:
        config = json.load(f)

    # Paths
    data_path = config['data_path']
    out_path = config['out_path']
    dir_name = config['unique_name']
    self_path = config['self_path']
    feil_path = config['feil_path']

    # Data specifications. If lemma_column is none, lemmatization will be computed.
    text_column = config['text_column']
    lemma_column = config['lemma_column']
    number_of_samples = config['number_of_samples']
    
    # Speed ups
    run_parallel = config['run_parallel']
    batch_size = config['batch_size']
    fast_scores = config['fast_scores']

    # Save results & see progress
    write_to_file = config['write_to_file']
    verbose = config['verbose']

    # Create directory for results
    if write_to_file:
        save_path = os.path.join(out_path, dir_name)
        os.mkdir(save_path)
        shutil.copy(args['config'], save_path)     
    
    if verbose:
        print('Hi! Starting score calculation pipeline...')
    
    # Load data, draw subsample, and drop rows with nan-values in the text
    df = read_data(data_path)

    if verbose:
        print(f'Your data looks like this: ')
        print(df.head(5))

    if number_of_samples != None:
        df = draw_random_sample(df, 42, number_of_samples)
        df = df.reset_index()
    df = df[df[[text_column]].isna().all(axis=1) == False]

    if verbose:
        print(f'After dropping empty text rows, the dataframe contains {len(df)} samples.')
        print('Successfully completed data loading!')

    # Lemmatization
    if not lemma_column:
        if verbose: 
            print('Starting lemmatization with the trankit pipeline. Get a cup of coffee, this might take a while...')
            start_trankit = time.perf_counter()

        lemma_column = 'lemmatized'
        df = preprocess_transcripts(df, text_column)
        trankit_pipe = Pipeline('finnish')
        text_filtered_list = df["transcript"]

        if run_parallel:
            df = parallel_lemmatizer(text_filtered_list, lemma_column, num_workers=(cpu_count()-1), batch_size=batch_size)
        else:
            text_lemmatized = []
            for i, text_sample in enumerate(text_filtered_list):
                if type(text_sample) != str:
                    if pd.isna(text_sample):
                        text_lemmatized.append("")
                        continue
                lemmatization = apply_trankit_pipeline(text_sample)
                text_lemmatized.append(lemmatization)
            df[lemma_column] = text_lemmatized
        
        if verbose:
            end_trankit = time.perf_counter()
            time_trankit = end_trankit - start_trankit
            print(f'Successfully completed lemmatization with the Trankit pipeline! ' 
                  f'That took {np.around((time_trankit)/60, decimals=3)} minutes.')
        
    # sentiment lexica
    if verbose:
        print(f'Starting score calculation. This might take a while...')
        start_score_cal = time.perf_counter()
    self = read_data(self_path, sep='\t')
    feil = read_data(feil_path, sep='\t')
    df.loc[:, 'utterance_scores'] = df[lemma_column].apply(calculate_utterance_score)

    # mappings
    if fast_scores:
        df.loc[:, 'valence_label_disc'] = df['utterance_scores'].apply(map_to_valence_disc_fast)
    else:
        df.loc[:, 'valence_label_disc'] = df[lemma_column].apply(lambda x: map_to_valence(x, continuous=False))

    if verbose:
        end_score_cal = time.perf_counter()
        time_score_cal = end_score_cal - start_score_cal
        print(f'Successfully completed score calculation. That took {np.around((time_score_cal)/60, decimals=3)} minutes.')
   
    # save data
    if write_to_file:
        df.to_pickle(os.path.join(save_path, f'scores_df.pkl')) 
    
    if verbose:
        if write_to_file:
            print(f'Please find all the results in the directory {save_path}.') 
        print('End of code reached. Goodbye and good luck with your research!')


    
