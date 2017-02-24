import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import StratifiedKFold
from time import time
	

def dcg_at_k(r, k, method=0):
	""" Returns discounted cumulative gain based on first k relevance scores (r)
	
	Parameters
	----------
	r: list or np.array
		Relevance scores descending in order of prediction likelihood
	k: int
		Number of relevance scores to consider
	method: int
		If 0 then weights are [1.0, 1.0, 0.6309, 0.5, 0.4307, ...]
		If 1 then weights are [1.0, 0.6309, 0.5, 0.4307, ...]
	Returns
	----------
	dcg: float
		Discounted cumulative gain
	"""
	r = np.asfarray(r)[:k]
	if r.size:
		if method == 0:
			return r[0] + np.sum(r[1:] / np.log2(np.arange(2, r.size + 1)))
		elif method == 1:
			return np.sum(r / np.log2(np.arange(2, r.size + 2)))
		else:
			raise ValueError('method must be 0 or 1.')
	return 0.


def ndcg_at_k(r, k, method=0):
	""" Returns normalized discounted cumulative gain based on first k relevance scores (r)
	
	Parameters
	----------
	r: list or np.array
		Relevance scores descending in order of prediction likelihood
	k: int
		Number of relevance scores to consider
	method: int
		If 0 then weights are [1.0, 1.0, 0.6309, 0.5, 0.4307, ...]
		If 1 then weights are [1.0, 0.6309, 0.5, 0.4307, ...]
	Returns
	----------
	ndcg: float
		Normalized discounted cumulative gain
	"""
	dcg_max = dcg_at_k(sorted(r, reverse=True), k, method)
	if not dcg_max:
		return 0.
	return dcg_at_k(r, k, method) / dcg_max
	
	
def score_predictions(preds, truth, n_modes=5):
	""" For each observation compares ranked list of predictions with true value 
		and estimates normalized discounted cumulative gain. 

	Parameters
	----------
	preds: pd.DataFrame
		Rows correspond to observations, columns correspond to predictions.
		Columns are sorted from left to right descending in order of prediction likelihood.
	truth: pd.Series
		True values of target variable, rows correspond to observations.
	n_modes: int
		Number of most probable predictions to consider

	Returns
	----------
	scores: list of length len(preds)
		A list of scores calculated as normalized discounted cumulative gain

	"""
	if (len(preds) != len(truth)):
		raise Exception("number of prediction cases must be the same")
	scores = []
	for i in range(len(preds)):
		r =  [1 if p == truth[i] else 0 for p in preds[i]] 
		scores.append(ndcg_at_k(r, n_modes, method=1))
	return scores
	
def split_sample_on_two(ds, perc, rs):
	n = len(ds)
	n_first = int(perc * n)
	sampler = np.arange(n)
	rs.shuffle(sampler)
	ds_first = ds.take(sampler[:n_first])
	ds_second = ds.take(sampler[n_first:])
	ds_first.index = range(len(ds_first))
	ds_second.index = range(len(ds_second))
	return ds_first, ds_second

	
def cross_validation_ndcg_score(estimator, df, resp_var, num_cv, k, rs = np.random.RandomState()):
	""" Evaluate ndcg scores for cross validation
	
	Parameters
    ----------
    estimator : estimator object implementing 'fit'
        The object to use to fit the data.
    df : data frame
        The data set with predicting and response variables. 
    resp_var : string 
        Name of response variable in dataframe df  
	k:  int	
		Number of predictions to consider for each test case. It is used in ndcg calculation.
	rs: numpy.random.RandomState object
	Returns
	---------
	scores : array of float, length = num_cv
        Array of scores of the estimator for each run of the cross validation.
	
	"""
	scores = []
	for i in range(num_cv):
		t0 = time()
		#split data set randomly
 		df_train, df_cv = split_sample_on_two(df, 0.7, rs)
		print "%s data items splited into %s train and %s cv items" % (len(df), len(df_train), len(df_cv))
		
		X_train = df_train.drop(resp_var, axis = 1).values
		X_cv = df_cv.drop(resp_var, axis = 1).values

		le = LabelEncoder()
		y_train = le.fit_transform(df_train[resp_var].values)
		y_cv = le.fit_transform(df_cv[resp_var].values)
	
		estimator.fit(X_train, y_train)
		pred_prob = estimator.predict_proba(X_cv)
		
		pred = []
		for i in range(len(pred_prob)):
			pred.append(np.argsort(pred_prob[i])[::-1])
		
		score = np.mean(score_predictions(pred, y_cv))
		scores.append(score)
		print "Calculated in {} min, {} sec".format(*divmod(time()-t0, 60))

		
	return scores
		
def strat_cross_validation_ndcg_score(estimator, df, resp_var, num_cv, k, rs = np.random.RandomState()):
	""" Evaluate ndcg scores for cross validation
	
	Parameters
    ----------
    estimator : estimator object implementing 'fit'
        The object to use to fit the data.
    df : data frame
        The data set with predicting and response variables. 
    resp_var : string 
        Name of response variable in dataframe df  
	k:  int	
		Number of predictions to consider for each test case. It is used in ndcg calculation.
	rs: numpy.random.RandomState object
	Returns
	---------
	scores : array of float, length = num_cv
        Array of scores of the estimator for each run of the cross validation.
	
	"""
	scores = []
	skf = StratifiedKFold(n_splits=num_cv, random_state=rs )

	for train, test in skf.split(df, df[resp_var]):		
		X_train = train.drop(resp_var, axis = 1).values
		X_test = test.drop(resp_var, axis = 1).values

		le = LabelEncoder()
		y_train = le.fit_transform(train[resp_var].values)
		y_test = le.fit_transform(test[resp_var].values)
	
		estimator.fit(X_train, y_train)
		pred_prob = estimator.predict_proba(X_test)
		
		pred = []
		for i in range(len(pred_prob)):
			pred.append(np.argsort(pred_prob[i])[::-1])
		
		score = np.mean(score_predictions(pred, y_test))
		scores.append(score)
		
	return scores


	
	

	
	
		