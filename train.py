#!/usr/bin/env python
# Train the adoption model and save it to model.bin.
# The steps mirror the notebook; see austin_animal_center_EDA.ipynb for the reasoning.

import pickle

import numpy as np
import pandas as pd

from sklearn.feature_extraction import DictVectorizer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score


# parameters

LEARNING_RATE = 0.05
MAX_LEAF_NODES = 15
MIN_TRAIN_ROWS = 200
OUTPUT_FILE = 'model.bin'

KEEP_ANIMALS = ['Dog', 'Cat']
DROP_INTAKES = ['Wildlife', 'Public Assist', 'Euthanasia Request']
DROP_OUTCOMES = ['Missing', 'Stolen', 'Lost']

DAYS_PER_UNIT = {'day': 1, 'week': 7, 'month': 30, 'year': 365}

# Names the shelter system truncated -> full breed name
BREED_FIXES = {
    'Anatol Shepherd': 'Anatolian Shepherd',
    'Bedlington Terr': 'Bedlington Terrier',
    'Bouv Flandres': 'Bouvier Des Flandres',
    'Boykin Span': 'Boykin Spaniel',
    'Bruss Griffon': 'Brussels Griffon',
    'Cavalier Span': 'Cavalier King Charles Spaniel',
    'Chesa Bay Retr': 'Chesapeake Bay Retriever',
    'Dachshund Stan': 'Dachshund',
    'Doberman Pinsch': 'Doberman Pinscher',
    'Eng Toy Spaniel': 'English Toy Spaniel',
    'Oriental Sh': 'Oriental Shorthair',
    'Patterdale Terr': 'Patterdale Terrier',
    'Pbgv': 'Petit Basset Griffon Vendeen',
    'Port Water Dog': 'Portuguese Water Dog',
    'Rhod Ridgeback': 'Rhodesian Ridgeback',
    'Sealyham Terr': 'Sealyham Terrier',
    'Sussex Span': 'Sussex Spaniel',
    'Treeing Tennesse Brindle': 'Treeing Tennessee Brindle',
    'West Highland': 'West Highland White Terrier',
    'Yorkshire': 'Yorkshire Terrier',
}


# features (also used by predict.py)

def parse_primary_breed(breed):
    breed = breed.str.replace('Black/Tan Hound', 'Black and Tan Hound', regex=False)
    return breed.str.removesuffix(' Mix').str.split('/').str[0].replace(BREED_FIXES)


def prepare_features(df, groups):
    """Raw intake records -> model features.

    `groups` holds the breeds, colors and conditions common enough in training
    to keep; anything else becomes 'Other'.
    """
    breed = df['breed'].str.replace('Black/Tan Hound', 'Black and Tan Hound', regex=False)
    primary_breed = parse_primary_breed(df['breed'])
    color_primary = df['color'].str.split('/').str[0]
    sex = df['sex_upon_intake'].fillna('Unknown')
    age = df['age_upon_intake'].str.split()

    return pd.DataFrame({
        'animal_type': df['animal_type'],
        'intake_type': df['intake_type'],
        'intake_condition': df['intake_condition'].where(
            df['intake_condition'].isin(groups['intake_condition']), 'Other'),
        'sex': sex.str.split().str[-1],
        'breed_type': np.select(
            [breed.str.endswith(' Mix'), breed.str.contains('/')], ['mix', 'cross'], default='pure'),
        'primary_breed': primary_breed.where(
            primary_breed.isin(groups['primary_breed']), 'Other ' + df['animal_type']),
        'color_primary': color_primary.where(
            color_primary.isin(groups['color_primary']), 'Other'),
        'age_days': (age.str[0].astype(float) * age.str[1].str.rstrip('s').map(DAYS_PER_UNIT)).clip(lower=0),
        'is_fixed': sex.str.contains('Neutered|Spayed').astype(int),
        'is_multicolor': df['color'].str.contains('/').astype(int),
        'intake_month': pd.to_datetime(df['intake_datetime']).dt.month,
        'prior_stays': df['prior_stays'],
    })


def find_common_values(df_train):
    """Values seen at least MIN_TRAIN_ROWS times in training; the rest become 'Other'."""
    def common(values):
        counts = values.value_counts()
        return set(counts[counts >= MIN_TRAIN_ROWS].index)

    return {
        'primary_breed': common(parse_primary_breed(df_train['breed'])),
        'color_primary': common(df_train['color'].str.split('/').str[0]),
        'intake_condition': common(df_train['intake_condition']),
    }


# data preparation

def load_stays():
    """One row per shelter stay: each intake matched to its outcome."""
    intakes = pd.read_csv('data/aac_intakes.csv')
    outcomes = pd.read_csv('data/aac_outcomes.csv')

    intakes.columns = intakes.columns.str.replace(' ', '_').str.lower()
    outcomes.columns = outcomes.columns.str.replace(' ', '_').str.lower()
    intakes = intakes.drop_duplicates()
    outcomes = outcomes.drop_duplicates()

    intakes['intake_datetime'] = pd.to_datetime(intakes['datetime'])
    outcomes['outcome_datetime'] = pd.to_datetime(outcomes['datetime'].str[:19], format='%Y-%m-%dT%H:%M:%S')

    # Date-only outcomes count as end of that day
    date_only = outcomes['datetime'].str.len() > 19
    outcomes['match_time'] = outcomes['outcome_datetime'].where(
        ~date_only, outcomes['outcome_datetime'] + pd.Timedelta(hours=23, minutes=59))

    # Match each intake to the animal's first outcome after it,
    # unless that outcome belongs to the animal's next stay
    intakes = intakes.sort_values('intake_datetime')
    intakes['next_intake_datetime'] = intakes.groupby('animal_id')['intake_datetime'].shift(-1)

    df = pd.merge_asof(
        intakes,
        outcomes.sort_values('match_time'),
        left_on='intake_datetime',
        right_on='match_time',
        by='animal_id',
        direction='forward',
        suffixes=('_intake', '_outcome'),
    )
    df = df[(df['match_time'] < df['next_intake_datetime']) | df['next_intake_datetime'].isna()]
    df = df[df['outcome_type'].notna()]

    df = df.rename(columns={
        'animal_type_intake': 'animal_type',
        'breed_intake': 'breed',
        'color_intake': 'color',
    })

    df = (
        df
        .query('animal_type in @KEEP_ANIMALS')
        .query('intake_type not in @DROP_INTAKES')
        .query('outcome_type not in @DROP_OUTCOMES')
        .assign(adopted=lambda d: (d['outcome_type'] == 'Adoption').astype(int))
        .sort_values('intake_datetime')
        .reset_index(drop=True)
    )
    df['prior_stays'] = df.groupby('animal_id').cumcount()
    return df


# training

def train(df_train, groups):
    dicts = prepare_features(df_train, groups).to_dict(orient='records')

    pipeline = make_pipeline(
        DictVectorizer(sparse=False),
        HistGradientBoostingClassifier(
            learning_rate=LEARNING_RATE,
            max_leaf_nodes=MAX_LEAF_NODES,
            max_iter=500,
            early_stopping=True,
            random_state=1,
        ),
    )
    pipeline.fit(dicts, df_train['adopted'])
    return pipeline


def predict(df, pipeline, groups):
    dicts = prepare_features(df, groups).to_dict(orient='records')
    return pipeline.predict_proba(dicts)[:, 1]


if __name__ == '__main__':
    print('loading data')
    df = load_stays()

    # Time-based split: oldest 60% train, next 20% validation, newest 20% test
    n = len(df)
    df_train = df.iloc[:int(0.6 * n)]
    df_val = df.iloc[int(0.6 * n):int(0.8 * n)]
    df_test = df.iloc[int(0.8 * n):]

    groups = find_common_values(df_train)

    print('validation')
    pipeline = train(df_train, groups)
    auc = roc_auc_score(df_val['adopted'], predict(df_val, pipeline, groups))
    print(f'validation auc={auc:.4f}')

    print('training the final model on train + validation')
    df_full_train = pd.concat([df_train, df_val])
    pipeline = train(df_full_train, groups)
    auc = roc_auc_score(df_test['adopted'], predict(df_test, pipeline, groups))
    print(f'test auc={auc:.4f}')

    with open(OUTPUT_FILE, 'wb') as f_out:
        pickle.dump((pipeline, groups), f_out)

    print(f'the model is saved to {OUTPUT_FILE}')
