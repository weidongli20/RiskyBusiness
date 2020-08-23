# RiskyBusiness
Build and evaluate several machine-learning models to predict credit risk using free data from LendingClub

![Credit Risk](Images/credit-risk.jpg)

## Background

Auto loans, mortgages, student loans, debt consolidation ... these are just a few examples of credit and loans that people are seeking online. Peer-to-peer lending services such as LendingClub or Prosper allow investors to loan other people money without the use of a bank. However, investors always want to mitigate risk, so you have been asked by a client to help them use machine learning techniques to predict credit risk.

In this assignment, we will build and evaluate several machine-learning models to predict credit risk using free data from LendingClub. Credit risk is an inherently imbalanced classification problem (the number of good loans is much larger than the number of at-risk loans), so we will need to employ different techniques for training and evaluating models with imbalanced classes. You will use the imbalanced-learn and Scikit-learn libraries to build and evaluate models using the two following techniques:

1. [Resampling](#Resampling)
2. [Ensemble Learning](#Ensemble-Learning)

#### Resampling

We will use the [imbalanced learn](https://imbalanced-learn.readthedocs.io) library to resample the LendingClub data and build and evaluate logistic regression classifiers using the resampled data.

We will:

1. Oversample the data using the `Naive Random Oversampler` and `SMOTE` algorithms.
2. Undersample the data using the `Cluster Centroids` algorithm.
3. Over- and under-sample using a combination `SMOTEENN` algorithm.

For each of the above, we will need to:

1. Train a `logistic regression classifier` from `sklearn.linear_model` using the resampled data.
2. Calculate the `balanced accuracy score` from `sklearn.metrics`.
3. Calculate the `confusion matrix` from `sklearn.metrics`.
4. Print the `imbalanced classification report` from `imblearn.metrics`.

Use the above to answer the following:

> Which model had the best balanced accuracy score?
> 
>      Naive Random Oversampling:      pre/rec 0.99/0.64
>      SMOTE Oversampling:             pre/rec 0.99/0.65
>      ClusterCentroids Undersampling: pre/rec 0.99/0.39
>      Combined SMOTEENN:              pre/rec 0.99/0.54
       
> Which model had the best recall score?
>
>      SMOTE Oversampling had the best recall score, 0.65
      
> Which model had the best geometric mean score?
>
>      Oversampling (Random and SMOTE) had the best geometric mean score, 0.65
      

#### Ensemble Learning

In this section, we will train and compare two different ensemble classifiers to predict loan risk and evaluate each model. we will use the `balanced random forest classifier` and the `easy ensemble AdaBoost classifier`.

Be sure to complete the following steps for each model:

1. Train the model using the quarterly data from LendingClub provided in the `Resource` folder.
2. Calculate the balanced accuracy score from `sklearn.metrics`.
3. Print the confusion matrix from `sklearn.metrics`.
4. Generate a classification report using the `imbalanced_classification_report` from imbalanced learn.
5. For the balanced random forest classifier only, print the feature importance sorted in descending order (most important feature to least important) along with the feature score.

Use the above to answer the following:

> Which model had the best balanced accuracy score?
>     
>     RandomForestClassifier had the best balanced accruacy score
>      
     
> Which model had the best recall score?
>     
>     RandomForestClassifier had the best balanced accruacy score
>      
      
> Which model had the best geometric mean score?
>   
>     EasyEnsembleClassifier had the best geometric mean score
>      
      
> What are the top three features?
>
>     total_rec_prncp: (0.08293057894907066)
>     total_rec_int: (0.06932938660179051)
>     last_pymnt_amnt: (0.06644736065367392)