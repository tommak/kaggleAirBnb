# **AirBnb New User Bookings**

As a part of [Kaggle competition](https://www.kaggle.com/c/airbnb-recruiting-new-user-bookings) AirBnb provided a list of users along with their demographics, web session records, and some summary statistics. The goal of the project was to predict in which country a new user will make his or her first booking.

Described solution got 160th place being in top 11% among all participants and got 0.88484 score on unseen test data, profile can be found [here](https://www.kaggle.com/tmakarova).

**Environment Setup**

All used packages are listed in .yml file. Project uses external python library xgboost.
Those who use Anaconda can use provided _conda_env.yml_ filr and create new environment with
```
$ conda env create -f environment.yml
```
However this will install quite old xgboost version. Code will be run correctly, but some features from analysis (plotting of feature importance) will be missed. In order to get newer version of xgboost, one can follow this [instructions](http://xgboost.readthedocs.io/en/latest/build.html).

**Data**
Data can be downloaded from [here]( [https://www.kaggle.com/c/airbnb-recruiting-new-user-bookings/data](https://www.kaggle.com/c/airbnb-recruiting-new-user-bookings/data)).
Please note that file with train users data (&#39;train\_users\_2.csv&#39;) should be named users.csv, test users (data for submission) should be named &#39;submission\_users.csv&#39;.

**Code**
_par\_variator.py_
Calculate cross validation results for specified classifier and range of parameters.

_run\_model.py_
Train specified model on train data set and run it on evaluation data set or/and generate submission file. Result of evaluation step is a csv file containing for each case from evaluation data set a ranked lists of predictions along with true class value.