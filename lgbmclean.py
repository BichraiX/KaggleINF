import os
import re
import torch
import numpy as np
import pandas as pd
from transformers import BertTokenizer, BertModel
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import pickle
import lightgbm as lgb
import random
import optuna
import utils

# Hyperparameters
max_length = 44
MAX_TWEETS= 650
PAD_TWEET = "[PAD]" 

#Load preprocessed data 
df = utils.load_or_prepare_data_spacy()


bert_model = BertModel.from_pretrained('bert-base-uncased')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

#Padding function
def pad_or_truncate_tweets(tweets, max_tweets=MAX_TWEETS, pad_tweet=PAD_TWEET):
    """
    Limite les tweets à max_tweets et ajoute du padding si nécessaire.
    """
    random.shuffle(tweets)
    tweets = tweets[:max_tweets]  # Limiter à max_tweets
    padding_needed = max_tweets - len(tweets)  # Calculer le padding requis
    if padding_needed > 0:
        tweets += [pad_tweet] * padding_needed  # Ajouter des tweets de padding
    return tweets

# Group tweets and labels per MatchID and per PeriodID
grouped_tweets = (df.groupby(['MatchID', 'PeriodID'])['Tweet'].apply(lambda tweets: pad_or_truncate_tweets(tweets.tolist(), MAX_TWEETS, PAD_TWEET)).unstack(fill_value=[]))
grouped_labels = df.groupby(['MatchID', 'PeriodID'])['EventType'].max().unstack(fill_value=0)

def prepare_features_and_labels(grouped_tweets, grouped_labels, bert_model, tokenizer, output_file="features_labels.pkl"):
    """
    Prepare and save features and labels
    
    :param grouped_tweets: Grouped tweets per MatchID and PeriodID.
    :param grouped_labels: Grouped labels per MatchID and PeriodID.
    :param bert_model: Trained BERT model
    :param tokenizer: BERT tokenizer
    :param output_file: Name of the file to save the features
    :return: Tuple (features, labels)
    """
    #Loads the data if it already exists
    if os.path.exists(output_file):
        print(f"Chargement des features et labels depuis {output_file}...")
        with open(output_file, "rb") as f:
            data = pickle.load(f)
        return data["features"], data["labels"]
    
    labels = []
    features = []
    for match_id in grouped_tweets.index:  # Iterate over the MatchID (rows)
        for period_id in grouped_tweets.columns:
            tweets = grouped_tweets.loc[match_id, period_id]
            if not isinstance(tweets, list) or len(tweets) == 0:
                continue

            # Tokenisation
            feature = tokenizer(
                tweets,
                max_length=max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            embed = bert_model.embeddings.word_embeddings(feature['input_ids'])

            # Mean for each tweet
            features_mean = embed.mean(dim=1, keepdims=False, dtype=torch.float32)
            features.append(features_mean)
            labels.append(grouped_labels.loc[match_id, period_id])
        
    # Saving file
    with open(output_file, "wb") as f:
        pickle.dump({"features": features, "labels": labels}, f)

    return features, labels



features, labels = prepare_features_and_labels(grouped_tweets=grouped_tweets,grouped_labels=grouped_labels,bert_model=bert_model,tokenizer=tokenizer,output_file="features_labels_lgbm_spacy.pkl")

features = [f.detach().numpy() for f in features]  
features = np.array([f.mean(axis=0) for f in features]) # Mean per period
labels= np.array(labels) 

#Parameters optimization with Optuna

def optuna_optimization(num_trials, X, y, kfold, test_size):
    """
    
    :param num_trials: Number of Optuna trials
    :param X: Training features
    :param y: Training labels
    :param kfold: Fold number for cross-validation
    :param test_size: Test set proportion
    :return: Best parameters found
    """
    def objective(trial):
        # Hyperparameters
        param = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting_type': 'gbdt',
        'learning_rate': trial.suggest_float('learning_rate', 1e-4, 0.3, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 10, 1000),
        'max_depth': trial.suggest_int('max_depth', -1, 50),  
        'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 10, 200),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 1.0),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 1.0),
        'bagging_freq': trial.suggest_int('bagging_freq', 1, 10),
        'lambda_l1': trial.suggest_float('lambda_l1', 0, 5),
        'lambda_l2': trial.suggest_float('lambda_l2', 0, 5),
        'min_gain_to_split': trial.suggest_float('min_gain_to_split', 0.0, 1.0),
        'random_state': 42,
        'verbose': -1,
    }
        num_boost_round = trial.suggest_int('n_estimators', 100, 1000)
        # Cross validation
        scores = []
        for i in range(kfold):
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_size, random_state=42 + i)
            train_data = lgb.Dataset(X_train, label = y_train)
            test_data = lgb.Dataset(X_val,label= y_val, reference = train_data)   
            model = lgb.train(param,train_data,valid_sets=[train_data, test_data], num_boost_round=num_boost_round)            
            # Accuracy on the validation set
            y_pred = model.predict(X_val)
            y_pred = (y_pred > 0.5).astype(int)
            val_accuracy = accuracy_score(y_val, y_pred)
            scores.append(val_accuracy)

        # Accuracy maximisation
        return -np.mean(scores)

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=num_trials)
    print("Best Trial:")
    print(study.best_trial.params)
    print("Best Score:", -study.best_value)
    return study.best_trial.params


best_params = optuna_optimization(
    num_trials=1,
    X=features,
    y=labels,
    kfold=5,
    test_size=0.4
)

# Training the final model with the best parameters found
print("Training with best parameters...")

final_model = lgb.LGBMClassifier(**best_params)
final_model.fit(features, labels)


#Loading preprocessed evluation dataset for submission
df_eval = utils.load_or_prepare_data_eval_spacy()
grouped_tweets = (df_eval.groupby(['MatchID', 'PeriodID'])['Tweet'].apply(lambda tweets: pad_or_truncate_tweets(tweets.tolist(), MAX_TWEETS, PAD_TWEET)).unstack(fill_value=[]))
ids = df_eval.groupby(['MatchID', 'PeriodID'])['Tweet'].apply(lambda tweets: pad_or_truncate_tweets(tweets.tolist(), MAX_TWEETS, PAD_TWEET)).index.to_list() #for submission


def prepare_features(grouped_tweets, bert_model, tokenizer, output_file):
    """
    Prepare and save features and labels
    
    :param grouped_tweets: Grouped tweets per MatchID and PeriodID.
    :param bert_model: Trained BERT model
    :param tokenizer: BERT tokenizer
    :param output_file: Name of the file to save the features
    :return: features
    """
    if os.path.exists(output_file):
        print(f"Chargement des features et labels depuis {output_file}...")
        with open(output_file, "rb") as f:
            data = pickle.load(f)
        return data["features"]
    features = []
    for match_id in grouped_tweets.index: 
        for period_id in grouped_tweets.columns:
            tweets = grouped_tweets.loc[match_id, period_id]
            if not isinstance(tweets, list) or len(tweets) == 0:
                continue

           
            feature = tokenizer(
                tweets,
                max_length=max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            embed = bert_model.embeddings.word_embeddings(feature['input_ids'])
            features_mean = embed.mean(dim=1, keepdims=False, dtype=torch.float32)
            features.append(features_mean)
    print(f"Sauvegarde des features et labels dans {output_file}...")
    with open(output_file, "wb") as f:
        pickle.dump({"features": features}, f)

    return features

features_eval = prepare_features(
    grouped_tweets=grouped_tweets,
    bert_model=bert_model,
    tokenizer=tokenizer,
    output_file="features_labels_eval_spacy.pkl"
)
features_eval = [f.detach().numpy() for f in features_eval]  
features_eval = np.array([f.mean(axis=0) for f in features_eval])
pred_eval= final_model.predict(features_eval)
output_df = pd.DataFrame({
    'ID': [f"{match_id}_{period_id}" for match_id, period_id in ids],  
    'Prediction': pred_eval
})

output_df.to_csv("submission.csv", index=False)