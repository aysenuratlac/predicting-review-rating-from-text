
import sys
sys.path.append('..')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from pathlib import Path
import json
from datetime import datetime
import time
import pickle

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_validate, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.base import clone

from scipy import stats

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    make_scorer
)

from src.tokenizers.syllable_tokenizer import TurkishSyllableTokenizer
from src.tokenizers.word_tokenizer import TurkishWordTokenizer
from src.tokenizers.zeyrek_word_tokenizer import TurkishWordZeyrekTokenizer

import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--model_type", type=str)
parser.add_argument("--vec_type", type=str)
parser.add_argument("--token_type", type=str)
parser.add_argument("--exp", type=str)

parser.add_argument(
    "--downsample",
    action="store_true",
    default=False
)

parser.add_argument(
    "--class_weight",
    type=str,
    default=None
)

args = parser.parse_args()

MODEL_TYPE = args.model_type
VEC_TYPE = args.vec_type
TOKEN_TYPE = args.token_type
DOWNSAMPLE = args.downsample
CLASS_WEIGHT = args.class_weight
EXP = args.exp
NAME = "{}_{}_{}_{}".format(MODEL_TYPE, VEC_TYPE, TOKEN_TYPE, EXP)

# Download NLTK resources (required for word-based tokenizers)
import nltk
try:
    nltk.data.find('tokenizers/punkt_tab')
    print('NLTK punkt_tab exist')
except LookupError:
    print('NLTK punkt_tab downloads...')
    nltk.download('punkt_tab', quiet=True)
    print('NLTK punkt_tab downloaded!')

from scipy.stats import binomtest, norm, chisquare


warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

exp_name = NAME
exp_name_r = r"{}".format(NAME)

# Results path
RESULTS_DIR = Path(exp_name_r)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

log_file = open(f"/okyanus/users/aatlac/final_experiments/{exp_name}/report.log","w")
sys.stdout=log_file


print("="*80)
print("TURKISH NLP CLASSIFICATION - EXPERIMENT PIPELINE")
print("="*80)
print(f"Random seed: {RANDOM_SEED}")
print(f"Main results folder: {RESULTS_DIR}")
print("="*80)


# ## 2. Data Uplaod

df1 = pd.read_csv('../data/raw/public_dataset.csv', low_memory=False)

df = df1[["review_text", "review_rating"]]
df = df.dropna()

print("Dataset Info")
print("="*60)
print(f"Total row: {len(df):,}")
print(f"Total column: {len(df.columns)}")
print(f"Columns: review_text, review_rating")

# Class Distribution
print("\nClass Distribution (Rating)")
print("="*60)
rating_counts = df['review_rating'].value_counts().sort_index()
print(rating_counts)
print(f"Total: {rating_counts.sum():,}")

# text length analysis
df['text_length'] = df['review_text'].astype(str).str.len()
print(f"\nText length Statistics")
print("="*60)
print(df['text_length'].describe())


# ## 3. Outlier function

def remove_outliers_percentile(df, column='text_length', lower_pct=2.5, upper_pct=97.5):
    """
    Removes outliers
    
    Parameters:
    -----------
    df : DataFrame
    column : str
    lower_pct : float
    upper_pct : float
    
    Returns:
    --------
    DataFrame : cleaned dataset
    dict : Statistics
    """
    lower_bound = df[column].quantile(lower_pct / 100)
    upper_bound = df[column].quantile(upper_pct / 100)
    
    df_cleaned = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)].copy()
    
    removed = len(df) - len(df_cleaned)
    
    stats_dict = {
        'lower_percentile': lower_pct,
        'upper_percentile': upper_pct,
        'lower_bound': float(lower_bound),
        'upper_bound': float(upper_bound),
        'original_size': len(df),
        'cleaned_size': len(df_cleaned),
        'removed_count': removed,
        'removed_percentage': float(removed / len(df) * 100)
    }
    
    return df_cleaned, stats_dict

# ## Class balancing (downsampling) helper
def balance_dataframe_by_rating(
    df: pd.DataFrame,
    label_col: str = "review_rating",
    classes=range(1, 6),
    random_seed: int = 42,
    verbose: bool = True
):


    # 1) Min. class size
    min_class_size = df[label_col].value_counts().min()

    if verbose:
        print(f"Minimum sınıf boyutu: {min_class_size:,}")
        print(f"Her sınıftan {min_class_size:,} örnek alınacak...\n")

    balanced_dfs = []

    # 2)each equal #of sample from each class
    for cls in classes:
        class_df = df[df[label_col] == cls]
        sampled = class_df.sample(
            n=min_class_size,
            random_state=random_seed
        )
        balanced_dfs.append(sampled)

        if verbose:
            print(f"{label_col} {cls}: {len(sampled):,} örnek")

    # 3) merge
    df_balanced = pd.concat(balanced_dfs, ignore_index=True)

    if verbose:
        print(f"\nTotal baalnced data: {len(df_balanced):,}")


    df_balanced = (
        df_balanced
        .sample(frac=1, random_state=random_seed)
        .reset_index(drop=True)
    )

    return df_balanced

# ## 4. Pipeline function

def create_pipeline(model_type, vectorizer_type, tokenizer_type, **model_params):
    """
    Creates classification pipeline
    
    Parameters:
    -----------
    model_type : str
        'logistic', 'knn', or 'svm'
    vectorizer_type : str
        'tfidf' or 'count'
    tokenizer_type : str
        'word', 'syllable','zeyrek' or 'char'
    **model_params : dict
    
    Returns:
    --------
    Pipeline : Scikit-learn pipeline object
    """
    # Tokenizer selection
    if tokenizer_type == 'word':
        tokenizer = TurkishWordTokenizer()
        tokenize_fn = tokenizer.tokenize
        analyzer = 'word'
    elif tokenizer_type == 'syllable':
        tokenizer = TurkishSyllableTokenizer()
        tokenize_fn = tokenizer.tokenize
        analyzer = 'word'
    elif tokenizer_type == 'char':
        tokenize_fn = None
        analyzer = 'char'
    elif tokenizer_type == 'zeyrek':
        tokenizer = TurkishWordZeyrekTokenizer()
        tokenize_fn = tokenizer.tokenize
        analyzer = 'word'
    else:
        raise ValueError(f"Geçersiz tokenizer_type: {tokenizer_type}")
    
    # Vectorizer selection
    if vectorizer_type == 'tfidf':
        if tokenizer_type == 'char':
            vectorizer = TfidfVectorizer(
                analyzer='char',
                ngram_range=(2, 4),
                max_features=10000
            )
        else:
            vectorizer = TfidfVectorizer(
                tokenizer=tokenize_fn,
                lowercase=False,
                max_features=10000
            )
    elif vectorizer_type == 'count':
        if tokenizer_type == 'char':
            vectorizer = CountVectorizer(
                analyzer='char',
                ngram_range=(2, 4),
                max_features=10000
            )
        else:
            vectorizer = CountVectorizer(
                tokenizer=tokenize_fn,
                lowercase=False,
                max_features=10000
            )
    else:
        raise ValueError(f"Geçersiz vectorizer_type: {vectorizer_type}")
    
    # Model selection
    if model_type == 'logistic':
        default_params = {
            'max_iter': 2000,
            'random_state': RANDOM_SEED,
            'n_jobs': -1,
                    }
        default_params.update(model_params)
        model = LogisticRegression(**default_params)
        
    elif model_type == 'knn':
        model_params = dict(model_params)
        model_params.pop('class_weight', None)  # KNN not supported

        default_params = {
            'n_neighbors': 5,
            'n_jobs': -1
        }
        default_params.update(model_params)
        model = KNeighborsClassifier(**default_params)

    elif model_type == 'svm':
        default_params = {
            'kernel': 'linear',
            'random_state': RANDOM_SEED,
            'max_iter': 2000,
                    }
        default_params.update(model_params)
        model = SVC(**default_params)
    else:
        raise ValueError(f"Geçersiz model_type: {model_type}")
    
    # Create Pipeline
    pipeline = Pipeline([
        ('vectorizer', vectorizer),
        ('classifier', model)
    ])
    
    return pipeline



def evaluate_with_cv(model, X, y, cv=5, random_state=None):
    """
    K-fold cross validation ile modeli değerlendirir.
    
    Parameters:
    -----------
    model : estimator
    X : array-like
    y : array-like
    cv : int
    random_state : int or None
        Random seed
    
    Returns:
    --------
    dict : scores for each 
    """
    # Scoring metrics
    scoring = {
        'accuracy': make_scorer(accuracy_score),
        'f1_macro': make_scorer(f1_score, average='macro', zero_division=0),
        'f1_weighted': make_scorer(f1_score, average='weighted', zero_division=0)
    }
    
    # Cross validation splitter (deterministic + stratified)
    if isinstance(cv, int):
        seed = RANDOM_SEED if random_state is None else random_state
        cv_splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=seed)
    else:
        cv_splitter = cv

    # Cross validation
    cv_results = cross_validate(
        model, X, y,
        cv=cv_splitter,
        scoring=scoring,
        return_train_score=True,
        n_jobs=-1
    )
    
    # Arrange results
    results = {}
    for metric in ['accuracy', 'f1_macro', 'f1_weighted']:
        test_scores = cv_results[f'test_{metric}']
        train_scores = cv_results[f'train_{metric}']
        
        #Calculate 95% CI (t-distribution)
        n = len(test_scores)
        mean = np.mean(test_scores)
        std = np.std(test_scores, ddof=1)
        se = std / np.sqrt(n)
        ci = stats.t.interval(0.95, n-1, loc=mean, scale=se)
        
        results[metric] = {
            'test_mean': float(round(mean, 4)),
            'test_std': float(round(std, 4)),
            'test_ci_lower': float(round(ci[0], 4)),
            'test_ci_upper': float(round(ci[1], 4)),
            'test_scores': [float(round(x, 4)) for x in test_scores],
            'train_mean': float(round(np.mean(train_scores), 4)),
            'train_std': float(round(np.std(train_scores, ddof=1), 4)),
            'overfitting_gap': float(round(np.mean(train_scores) - mean, 4))
        }
    
    return results


def format_cv_results_2dp_for_json(cv_results: dict) -> dict:
    """Convert CV results to a JSON-friendly structure with 4-decimal strings (e.g., '0.1234')."""
    out = {}
    for metric, vals in cv_results.items():
        if not isinstance(vals, dict):
            out[metric] = vals
            continue
        out_metric = {}
        for k, v in vals.items():
            if isinstance(v, list):
                out_metric[k] = [f"{float(x):.4f}" for x in v]
            elif isinstance(v, (int, float, np.floating)):
                out_metric[k] = f"{float(v):.4f}"
            else:
                out_metric[k] = v
        out[metric] = out_metric
    return out


# ## Scenario runner (preprocessing + CV evaluation + saving outputs)

def run_scenario_with_full_output(scenario, df, base_results_dir):

    scenario_name = scenario['name']
    
    print(f"# SENARYO: {scenario_name}")
    
    # create scenario directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    scenario_dir = base_results_dir / f"{scenario_name}_{timestamp}"
    scenario_dir.mkdir(exist_ok=True, parents=True)
    
    print(f"\nKlasör oluşturuldu: {scenario_dir}")
    
    # Data preparation
    print("\nData preparation...")
    if scenario["downsample"]:
        df_scenario =balance_dataframe_by_rating(df)
        print(f"Min class a downsample yapıldı!")
    else:
        df_scenario = df.copy()

    if scenario['remove_outliers']:
        df_scenario2, outlier_stats = remove_outliers_percentile(
            df_scenario,
            column='text_length',
            lower_pct=scenario.get('outlier_lower_pct', 2.5),
            upper_pct=scenario.get('outlier_upper_pct', 97.5)
        )
        print(f"  Outliers removed")
        print(f"    Original: {outlier_stats['original_size']:,}")
        print(f"    Cleaned: {outlier_stats['cleaned_size']:,}")
        print(f"    Removed: {outlier_stats['removed_count']:,} ({outlier_stats['removed_percentage']:.2f}%)")
    else:
        df_scenario2 = df_scenario.copy()
        outlier_stats = None
        print(f"  Outliers kept: {len(df_scenario2):,} örnek")
    
    # Train-test split
    X = df_scenario2['review_text'].values
    y = df_scenario2['review_rating'].values

    

    pipeline = create_pipeline(
        model_type=scenario['model_type'],
        vectorizer_type=scenario['vectorizer_type'],
        tokenizer_type=scenario['tokenizer_type'],
        **scenario.get('model_params', {})
    )
    print(f"  Model: {scenario['model_type']}")
    print(f"  Vectorizer: {scenario['vectorizer_type']}")
    print(f"  Tokenizer: {scenario['tokenizer_type']}")
    print(f"Downsample? {scenario['downsample']}")
    print(f"outlier? {scenario['remove_outliers']}")
    
    
    pipeline_cv = clone(pipeline)
    cv_splits = scenario.get('cv', 5)
    cv_results = evaluate_with_cv(pipeline_cv, X, y, cv=cv_splits, random_state=RANDOM_SEED)
    print(f"CV Accuracy: {cv_results['accuracy']['test_mean']:.4f} ± {cv_results['accuracy']['test_std']:.4f}")
    
    # Save
    print("\n Results saving...")
    
    # Scenario Parameters
    scenario_params = {
        'scenario_name': scenario_name,
        'timestamp': timestamp,
        'model_type': scenario['model_type'],
        'vectorizer_type': scenario['vectorizer_type'],
        'tokenizer_type': scenario['tokenizer_type'],
        'remove_outliers': scenario['remove_outliers'],
        "downsample": scenario['downsample'],
        'outlier_stats': outlier_stats,
        'random_seed': RANDOM_SEED
    }
    
    with open(scenario_dir / 'scenario_params.json', 'w', encoding='utf-8') as f:
        json.dump(scenario_params, f, indent=2, ensure_ascii=False)
    print(f"  ✓ scenario_params.json")
    

    # 7.5. CV Results JSON
    with open(scenario_dir / 'cv_results.json', 'w', encoding='utf-8') as f:
        json.dump(format_cv_results_2dp_for_json(cv_results), f, indent=2, ensure_ascii=False)
    print(f"  ✓ cv_results.json")
    
    
    print(f"\nScenario completed: {scenario_name}")
    print(f"   Folder: {scenario_dir}")
    
# ## 7. Define Scenarios

print("\nScenarios are defining...")

scenarios = [
    {
        'name': f'{NAME}_with_outliers',
        'model_type': MODEL_TYPE,
        'vectorizer_type': VEC_TYPE,
        'tokenizer_type': TOKEN_TYPE,
        'model_params': {'class_weight': CLASS_WEIGHT},
        'downsample': DOWNSAMPLE,
        'remove_outliers': False
    }
    ,
    {
        'name': f'{NAME}_without_outliers',
        'model_type': MODEL_TYPE,
        'vectorizer_type': VEC_TYPE,
        'tokenizer_type': TOKEN_TYPE,
        'model_params': {'class_weight': CLASS_WEIGHT},
        'downsample': DOWNSAMPLE,
        'remove_outliers': True,
        'outlier_lower_pct': 2.5,
        'outlier_upper_pct': 97.5
    }
]

print(f"Total {len(scenarios)} scenario completed:")
for i, scenario in enumerate(scenarios, 1):
    print(f"  {i:2d}. {scenario['name']}")


# ## Run All Scenarios

print("\n Senarios starting...\n")
print("="*80)

all_summaries = []

for i, scenario in enumerate(scenarios, 1):
    print(f"\n{'='*80}")
    print(f"Scenario {i}/{len(scenarios)}")
    print(f"{'='*80}")
    
    try:
        run_scenario_with_full_output(scenario, df, RESULTS_DIR)
    except Exception as e:
        print(f"\nError: Cannot run the experiment: {scenario['name']}")
        print(f"   Error: {str(e)}")


print(f"\n\n{'#'*80}")
print(f"# All scenarios completed")



log_file.close()
sys.stdout=sys.__stdout__

