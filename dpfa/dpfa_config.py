# the code is mainly copy pasted from my another repo:
# https://github.com/scott-pu-pennstate/dktt_light
import os
import time
__last_modified__ = time.time() - os.path.getmtime(__file__)


CV_NUM = 5
MAX_LEN = 200
HOME_DIR = os.path.dirname(
    os.path.dirname(__file__))
DATA_DIR = os.path.join(
    HOME_DIR,
    'data')
NAME_CONVENTION = '{}-cv-{}-{}.csv'
VERBOSE = 1

config = {}

config['assist2017'] = {
    'id_col': 'ITEST_id',
    'time_col': 'startTime',
    'prob_col': 'problemId',
    'skill_col': 'skill',
    'score_col': 'correct',
    'skill_sep': '~'
}

config['stat2011'] = {
    'id_col': 'Anon Student Id',
    'time_col': 'Time',
    'prob_col': 'problem',
    'skill_col': 'duplicated_problem',
    'score_col': 'Outcome',
    'skill_sep': '~'
}

config['synthetic5'] = {
    'id_col': 'ids',
    'time_col': 'time_seq',
    'prob_col': 'prob_seq',
    'skill_col': None,
    'score_col': 'score_seq',
    'skill_sep': None
}

config['nips2020'] = {
    'id_col': 'id',
    'time_col': 'DateAnswered',
    'prob_col': 'QuestionId',
    'skill_col': None,
    'score_col': 'IsCorrect',
    'skill_sep': None
}

nips_2020_params = [
    {
        'hidden_size': 32,
        'dropout': 0.2,
        'regulate_dot_product': True,
        'time_decay': True,
        'normalize_embedding': True,
    },

    {'batch_size': 500,
     'epochs': 20,
     'validation_split': 0.1},

    {
        'lr': 0.006,
        'smoothing': 0.1,
    }
]

assist_2017_params = [
    {
        'hidden_size': 16,
        'dropout': 0.2,
        'regulate_dot_product': True,
        'time_decay': False,
        'normalize_embedding': True,
    },

    {'batch_size': 512,
     'epochs': 150,
     'validation_split': 0.1},

    {
        'lr': 0.015,
        'smoothing': 0.1,
    }
]

stat_2011_params = [
    {
        'hidden_size': 16,
        'dropout': 0.2,
        'regulate_dot_product': True,
        'time_decay': False,
        'normalize_embedding': True,
    },

    {'batch_size': 512,
     'epochs': 200,
     'validation_split': 0.1},

    {
        'lr': 0.015,
        'smoothing': 0.1,
    }
]

synthetic_5_params = [
    {
        'hidden_size': 16,
        'dropout': 0.2,
        'regulate_dot_product': False,
        'time_decay': False,
        'normalize_embedding': False,
    },

    {'batch_size': 500,
     'epochs': 100,
     'validation_split': 0.1},

    {
        'lr': 0.16,
        'smoothing': 0.1,
    }
]
