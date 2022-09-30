# %%
import json
from pathlib import Path
import random
import warnings
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    make_scorer,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

# from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from imblearn.pipeline import Pipeline
from imblearn.under_sampling import RandomUnderSampler

RANDOM_STATE = 42
warnings.filterwarnings(action="ignore", category=ConvergenceWarning)  # doesn't work...


# %%
def quantiles(pointers_path="FakeNewsNet/dataset"):
    pointers_path = Path(pointers_path)

    dfs = []
    for label in ("fake", "real"):
        for site in ("gossipcop", "politifact"):
            pointers_df = pd.read_csv(
                pointers_path / f"{site}_{label}.csv", index_col="id"
            )
            pointers_df["site"] = site
            pointers_df["label"] = label
            dfs.append(pointers_df)

    all_pointers = pd.concat(dfs)
    counts = 1 + all_pointers["tweet_ids"].str.count("\t")
    counts = counts.fillna(0)
    print(counts.quantile(np.arange(0.05, 1, 0.05)))


# quantiles()


def create_dataset(
    n_tweets,
    random_state=RANDOM_STATE,
    pointers_path="FakeNewsNet/dataset",
    contents_path="FakeNewsNet/code/fakenewsnet_dataset",
    saves_path=".",
):

    pointers_path = Path(pointers_path)
    contents_path = Path(contents_path)
    saves_path = Path(".")

    res = []

    for label in ("fake", "real"):
        for site in ("gossipcop", "politifact"):

            pointers_df = pd.read_csv(
                pointers_path / f"{site}_{label}.csv", index_col="id"
            )
            pointers_df = pointers_df.dropna(subset=["tweet_ids"])
            pointers_df["tweet_ids"] = pointers_df["tweet_ids"].str.split("\t")

            # Sample n_tweets out of the all tweets
            random.seed(random_state)
            pointers_df["tweet_ids"] = pointers_df["tweet_ids"].map(
                lambda lst: random.sample(lst, n_tweets) if len(lst) > n_tweets else lst
            )
            random.seed(None)

            site_label_path = contents_path / site / label
            for article_id in pointers_df.index:
                tweets_path = site_label_path / article_id / "tweets"

                tweet_ids = []
                tweet_texts = []
                for tweet_id in pointers_df.at[article_id, "tweet_ids"]:
                    tweet_json = tweets_path / f"{tweet_id}.json"
                    if tweet_json.exists():
                        with open(tweet_json) as f:
                            tweet_data = json.load(f)
                        if tweet_data["lang"] != "en":
                            continue
                        tweet_ids.append(tweet_data["id_str"])
                        tweet_texts.append(tweet_data["text"])

                record = dict(
                    article_id=article_id,
                    article_title=pointers_df.at[article_id, "title"],
                    tweet_ids=tweet_ids,
                    tweet_texts=tweet_texts,
                    label=label,
                    site=site,
                )
                res.append(record)

    df = pd.DataFrame.from_records(res, index="article_id")

    # Remove records with no tweets
    has_tweets = df["tweet_texts"].map(len) != 0
    if not has_tweets.all():
        print(
            "Could not retrieve any tweets for the following:", *df.index[~has_tweets]
        )
        df = df[has_tweets]

    df.to_pickle(saves_path / f"dataset-tweets{n_tweets}-seed{random_state}.pkl")


# create_dataset(n_tweets=11)  # 25% quantile
# create_dataset(n_tweets=37)  # 50% quantile
# create_dataset(n_tweets=65)  # 75% quantile


def load_Xy(path, test_size=0.2):
    df = pd.read_pickle(path)

    # Concatenate everything into one big text.
    # We'll only use bag of words model so it's ok.
    df["text"] = df["article_title"] + "\n" + df["tweet_texts"].str.join("\n")

    # Replace all links with a dummy
    df["text"] = df["text"].str.replace(r"http\S+", "<LINK_REPLACEMENT>", regex=True)

    X_train, X_test, y_train, y_test, id_train, id_test = train_test_split(
        df["text"],
        df["label"],
        df.index,
        test_size=test_size,
        random_state=RANDOM_STATE,
    )

    return X_train, X_test, y_train, y_test, id_train, id_test


class Densifier(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return X.toarray()


class Unsqueezer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return np.expand_dims(X, 1)


class Squeezer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return np.squeeze(X, 1)


def evaluate_cv(cv, X_test, y_test, colorbar=False, ax=None):
    y_pred = cv.best_estimator_.predict(X_test)
    f1 = f1_score(y_test, y_pred, pos_label="fake")
    precision = precision_score(y_test, y_pred, pos_label="fake")
    recall = recall_score(y_test, y_pred, pos_label="fake")
    clf_name = type(cv.best_estimator_["clf"]).__name__
    params_str = ", ".join(
        f"{key.split('__', 1)[1]} = {val}" for key, val in cv.best_params_.items()
    )
    scores_str = f"F1 = {f1:.2f}, Precision = {precision:.2f}, Recall = {recall:.2f}"
    title = f"{clf_name}\n{params_str}\n{scores_str}"
    
    if ax is None:
        fig, ax = plt.subplots()
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred, colorbar=colorbar, ax=ax)
    # ax.set_title(title)
    ax.set_title(clf_name)
    


# %%
# Small test size because we will downsample the train_set
X_train, X_test, y_train, y_test, id_train, id_test = load_Xy(
    "dataset-tweets11-seed42.pkl", test_size=0.1
)
vectorizer = TfidfVectorizer(
    strip_accents="unicode",
    stop_words="english",  # stop_words pretty irrelevant for bag of words
    max_df=0.99,
    min_df=0.005,
    # max_features=None
)
f1_scorer = make_scorer(f1_score, pos_label="fake")

# All of the below can be replaced by a single Pipeline if the tail is the classifier,
# but to avoid repeated computation we precompute everything here.
# Also, random_state here is for reproducibility only
print("Train size (pre balancing):", len((y_train)))
# Downsample
X_train = Unsqueezer().fit_transform(X_train)
X_train, y_train = RandomUnderSampler(random_state=RANDOM_STATE).fit_resample(
    X_train, y_train
)
X_train = Squeezer().fit_transform(X_train)
print("Train size (post balancing):", len(y_train))
#%% Vectorize
X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)
print("Test size:", len(y_test))
print("Vocabulary size:", len(vectorizer.vocabulary_))


# %%
# TODO: Declare all estimators (pipes) and parameters in a dictionary and loop them over
# SVM, Neural Network, (Perceptron?), Logistic, Random Forest, (GMM?), KMeans, Adaboost, (MultinomialNB?)

config = {
    "svm": [
        Pipeline(steps=[("clf", SVC())]),
        {"clf__C": [0.1, 1.0, 10.0], "clf__kernel": ["linear", "rbf"]},
    ],
    "neural_network": [
        Pipeline(steps=[("clf", MLPClassifier())]),
        {
            "clf__hidden_layer_sizes": [(64,), (32, 16)],
            "clf__learning_rate_init": [1e-3, 1e-4],
        },
    ],
    "logistic_regression": [
        Pipeline(steps=[("clf", LogisticRegression())]),
        {"clf__C": [0.1, 1.0, 10.0]},
    ],
    "random_forest": [
        Pipeline(steps=[("clf", RandomForestClassifier())]),
        {"clf__n_estimators": [100, 200], "clf__max_depth": [None, 10, 30, 100]},
    ],
    "adaboost": [
        Pipeline(steps=[("clf", AdaBoostClassifier())]),
        {"clf__n_estimators": [100, 200], "clf__learning_rate": [0.5, 1.0, 2.0]},
    ],
    # 'knn': [
    #     Pipeline(steps=[('clf', KNeighborsClassifier())]),
    #     {
    #         'clf__n_neighbors': [5, 10],
    #         'clf__weights': ['uniform', 'distance']
    #     }
    # ]
}

config_svd = {
    'svm': [
        Pipeline(steps=[('svd', TruncatedSVD(100)), ('clf', SVC())]),
        {
            'clf__C': [0.1, 1., 10.],
            'clf__kernel': ['linear', 'rbf']
        }
    ],
    'neural_network': [
        Pipeline(steps=[('svd', TruncatedSVD(100)), ('clf', MLPClassifier())]),
        {
            'clf__hidden_layer_sizes': [(64,), (32, 16)],
            'clf__learning_rate_init': [1e-3, 1e-4]
        }
    ],
    'logistic_regression': [
        Pipeline(steps=[('svd', TruncatedSVD(100)), ('clf', LogisticRegression())]),
        {
            'clf__C': [0.1, 1., 10.]
        }
    ],
    'random_forest': [
        Pipeline(steps=[('svd', TruncatedSVD(100)), ('clf', RandomForestClassifier())]),
        {
            'clf__n_estimators': [100, 200],
            'clf__max_depth': [None, 10, 30, 100]
        }
    ],
    'adaboost': [
        Pipeline(steps=[('svd', TruncatedSVD(100)), ('clf', AdaBoostClassifier())]),
        {
            'clf__n_estimators': [100, 200],
            'clf__learning_rate': [0.5, 1.]
        }
    ],
    # 'knn': [
    #     Pipeline(steps=[('svd', TruncatedSVD(100)), ('clf', KNeighborsClassifier())]),
    #     {
    #         'clf__n_neighbors': [5, 10],
    #         'clf__weights': ['uniform', 'distance']
    #     }
    # ]
}


def train_everything(config):
    res = {}
    for key, (clf, grid) in config.items():
        # Mind the mutation of clf
        cv = GridSearchCV(
            clf,
            grid,
            cv=5,
            scoring=f1_scorer,
            error_score="raise",
            # verbose=2,
            n_jobs=-1,
        )
        cv.fit(X_train, y_train)
        # evaluate_cv(cv, X_test, y_test)
        res[key] = cv
    # plt.show()
    return res


def image_grid(results, suptitle):
    fig, axs = plt.subplots(nrows=5, figsize=(6.4, 10))
    for (key, cv), ax in zip(results.items(), axs.flat):
        evaluate_cv(cv, X_test, y_test, ax=ax)
    # for ax in axs.flat[len(results):]:
    #         ax.axis('off')
    for ax in axs.flat:
            ax.axis('off')
    fig.suptitle(suptitle, size='xx-large')
    fig.savefig(f'{suptitle}.pdf')


# Because our data is imbalanced we use F1 score
# ROC AUC is bad for imbalanced data https://stackoverflow.com/q/44172162
# print(roc_auc_score(y_test == 'fake', y_pred == 'fake'))

# %%
# !!!!!!!!!!! COMPUTATION INTENSIVE PART !!!!!!!!!
# res_undersampling = train_everything(config)
# res_undersampling_svd = train_everything(config_svd)

# with open('res_undersampling.pkl', 'wb') as f:
#     pickle.dump(res_undersampling, f)    
# with open('res_undersampling_svd.pkl', 'wb') as f:
#     pickle.dump(res_undersampling_svd, f)

with open('res_undersampling.pkl', 'rb') as f:
    res_undersampling = pickle.load(f)
with open('res_undersampling_svd.pkl', 'rb') as f:
    res_undersampling_svd = pickle.load(f)


# %%
image_grid(res_undersampling, 'Without SVD')
image_grid(res_undersampling_svd, 'With SVD')


# %%
