## This README describes what each file of the repo was used for.

# difftransformer.py

This is the PyTorch code for the architecture of our Differential Transformer - based classifier. It was largely inspired by Andrej Karpathy's nanoGPT (https://github.com/karpathy/nanoGPT) along with the YouTube video that goes with it : https://www.youtube.com/watch?v=kCc8FmEb1nY&t=5145s

# optuna_tuning.py

This is the Optuna trial code we used to tune the hyperparameters of our Differential Transformer. Unfortunately due to computational limitations, we did not get very satisfactory results with this method and couldn't tune our model appropriately.

# trainer.py

This is the code to setup Distributed Data Parallel (DDP) training to train our Differential Transformer Classifier.  This was largely inspired by PyTorch's DDP tutorial along with the associated repo : https://github.com/pytorch/examples

# launch_distributed_training.sh

This file was created to make launching DDP Training easier (just run bash launch_distributed_training.sh in the terminal). We only used 6 computers so as not to monopolize all the computers and only ran the code at night for fairness to other students.

# utils.py

This file contains the functions that were used often in our code, along with the preprocessed dataframe to prevent rerunning the preprocessing function everytime which took time.

# train_test.py

Code for training a model and testing it on a 20% of the dataset on a single machine.

# train.py

Code for training the model on the whole dataset on a single machine.

# sentiment.py

Code used to perform sentiment analysis on the dataset. Unfortunately this did not yield very good results and we did not end up using this feature in our best models.

# lgbm_train_test.ipynb

Notebook to train and test our LGBM model using our fulll feature engineering.

# lgbm_submission.ipynb

Notebook to train the model on the whole dataset and generate a Kaggle submission csv file.

# lgbmclean.py

Code to get the best hyperparameters with optuna and train a model with those parameters, generating a submission file at the end.