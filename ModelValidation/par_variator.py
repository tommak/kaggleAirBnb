import numpy as np
import pandas as pd	
import pickle as pkl
from time import time
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier
from  sklearn.ensemble import GradientBoostingClassifier
#from xgboost.sklearn import XGBClassifier
from xgboost import XGBClassifier

from  cleaning import clean_manage_kaggle, clean_manage_kaggle_v1, clean_manage_users_ses, \
					clean_manage_users_ses_ind, clean_manage_users_ses_ext, clean_manage_users_ses_FE, \
					clean_manage_users_ses_FE_1, prepare_data

from  cross_validation import cross_validation_ndcg_score
from preprocess import read_users_ses, split_users, users_ses_merge
import itertools

def calc_model_runs(data, target, clf, parameters, res_file):
	df_res = pd.DataFrame([], columns = parameters.keys() + ["score"])


	L = [[(k, val) for val in v] for k, v in parameters.items() ]

	for p in map(dict, itertools.product(*L)):
		t0 = time()
		print "Start traing of {} model".format(clf.__name__)
		rs = np.random.RandomState(24254362)
		estimator =  clf(**p) 
		scores = cross_validation_ndcg_score(estimator, data, target, num_cv=3, k=5, rs=rs)
		print scores
		t=time()-t0
		print "Total calculation time: {} min, {} sec".format(*divmod(t, 60))

		new_line = p
		new_line["score"] = np.mean(scores)
		df_res = df_res.append(new_line, ignore_index=True)
	

	print df_res
	df_res.to_csv(res_file, index=False)


if __name__ == "__main__":

	# Perform split on train and evaluation data sets
	split_users("../data/users.csv", "../data/train_users.csv", \
	             "../data/evaluation_users.csv", test_size=0.1)

	#Read train data, clean and prepare in a format for algorithm input
	df = read_users_ses("../data/train_users.csv", "../data/sessions.csv", users_ses_merge, 
	                     "../data/train_users_ses.csv", "../data/agg_action_info.csv") 
	# \
	# 	 .sample(frac=0.2).reset_index(drop=True)

	cl_df, num_features, cat_features = clean_manage_users_ses(df)
	algo_df = prepare_data(cl_df, num_features, cat_features)
	algo_df["country_destination"] = cl_df["country_destination"]	


	#Setup ranges for algorithm parameter variation 
	parameters ={
		"max_depth" : [3, 6, 8, 10],
		"n_estimators" : [90],
		"learning_rate" : [0.1],
		"subsample" : [0.5],
		"colsample_bytree" : [0.5],
		"objective" : ['multi:softprob'],
		"seed" : [0]
	}

	calc_model_runs(algo_df, "country_destination", XGBClassifier, parameters, "xgb_parvar_add.csv")

	# parameters ={
	# 	"max_depth" : [6],
	# 	"n_estimators" : [40],
	# 	"learning_rate" : [0.15],
	# 	"subsample" : [0.5]
		
	# }

	# calc_model_runs(algo_df, "country_destination", GradientBoostingClassifier, parameters, "test_algorithm_smpl_sklearn.csv")




	parameters ={
		"n_estimators" : [100, 200, 300, 400],
		"max_features" : [150]
	}

	calc_model_runs(algo_df, "country_destination", RandomForestClassifier, parameters, "rf_parvar_ext1.csv")

	parameters ={
		"n_estimators" : [400],
		"max_features" : [15, 50, 100, 150]
	}

	calc_model_runs(algo_df, "country_destination", RandomForestClassifier, parameters, "rf_parvar_ext2.csv")

	