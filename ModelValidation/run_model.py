import numpy as np
import pandas as pd
from math import log
from sklearn.preprocessing import LabelEncoder
import pickle 
from time import time

from cleaning import clean_manage_v1, clean_manage_kaggle, clean_manage_v2, clean_manage_kaggle_v1, \
					clean_manage_users_ses, clean_manage_users_ses_ext, prepare_data
from preprocess import split_users, read_users_ses, users_ses_merge, get_sample_weight
from cross_validation import score_predictions
import getopt, sys

try:
	from xgboost.sklearn import XGBClassifier
except ImportError:
	from xgboost import XGBClassifier


def load_or_fit(clf, X, y, path, dump_new=True, features_list=None, features_path=None, sample_weight=None):
	try:
		with open(path, "r") as clf_infile:
			fitted_clf = pickle.load(clf_infile)
		print "Classifier was loaded from ", path
	except IOError:
		print "Failed to load fitted classifier\nStart fitting..."
		t0 = time()
		fitted_clf = clf
		fitted_clf.fit(X, y, sample_weight=sample_weight)
		t = time() - t0
		print "Classifier fitted in {} min {} sec".format(*divmod(t, 60))
		if dump_new:
			with open(path, "w") as clf_outfile:
				pickle.dump(fitted_clf, clf_outfile)
			print "Fitted classifier was dumped to ", path

		if features_path:
			try:
				with open(features_path, "w") as feat_outfile:
					pickle.dump(features_list, feat_outfile)
			except NameError as e:
				print "If features_path is specified, features_list should be specified as well" 
				print e.message
	return fitted_clf 


def get_predictions(df_train, df_eval, clf, clf_path, class_weights=None):
	le_tr = LabelEncoder()
	tr_labels = df_train["country_destination"].values
	y = le_tr.fit_transform(tr_labels)

	df_train_eval = pd.concat((df_train, df_eval), axis=0, ignore_index=True)
		
	cl_df_train_eval, num_features, cat_features = clean_manage_users_ses(df_train_eval)
	cl_df_train_eval.drop("country_destination", axis=1, inplace=True)

	sample_weight = None
	if class_weights:
		sample_weight = get_sample_weight(class_weights, df_train["country_destination"])

	X_train_eval =  prepare_data(cl_df_train_eval, num_features, cat_features)
	features_list = X_train_eval.columns
	X_train_eval = X_train_eval.values
		
	piv_train = df_train.shape[0]                    
	X = X_train_eval[:piv_train]
	X_eval = X_train_eval[piv_train:]


	clf = load_or_fit(clf, X, y, clf_path, dump_new=True, features_list=features_list, \
											features_path="dev/features_1.pkl", sample_weight=sample_weight)
		
	y_pred = clf.predict_proba(X_eval) 
	return y_pred, le_tr

if __name__ == "__main__":

	split_users("../data/users.csv", "../data/train_users.csv", \
	             "../data/evaluation_users.csv", test_size=0.1)

	#Read train data, clean and prepare in a format for algorithm input
	df_train = read_users_ses("../data/train_users.csv", "../data/sessions.csv", users_ses_merge, 
	                     "../data/train_users_ses.csv", "../data/agg_action_info.csv")
	df_eval = read_users_ses("../data/evaluation_users.csv", "../data/sessions.csv", users_ses_merge, 
	                     "../data/evaluation_users_ses.csv")

	# sub_path = '../submissions/sub_test.csv'
	# eval_path = '../evaluations/eval_test.csv'

	try:
		options,_ = getopt.getopt(sys.argv[1:],"s:e:",["submission=", "evaluation="])
	except getopt.GetoptError as err:
		# print help information and exit:
		print err 
		sys.exit(2)

	if not options:
		# Set default options
		eval_path = '../evaluations/eval_clf_md8.csv'
		GEN_SUBMISSION = False
		VALIDATE = True
	else:
		GEN_SUBMISSION = False
		VALIDATE = False

	for opt, arg in options:
		if opt in ("-s", "--submission"):
			GEN_SUBMISSION = True
			sub_path = arg
		if opt in ("-e", "--evaluation"):
			VALIDATE = True
			eval_path = arg

	clf = XGBClassifier(max_depth=8, learning_rate=0.1, n_estimators=90,
	                    objective='multi:softprob', subsample=0.5, colsample_bytree=0.5, seed=0) 

	# class_weights = {country : 1./12 for country in df_train.country_destination.unique()}
	class_weights = None
	y_pred, lbl_encoder = get_predictions(df_train, df_eval, clf, "dev/clf_md8.pkl", class_weights=class_weights)


	if GEN_SUBMISSION:
		print "Generate submission and save to {}".format(sub_path)
		id_eval = df_eval['id']
		ids = []  #list of ids
		cts = []  #list of countries
		for i in range(len(id_eval)):
			idx = id_eval[i]
			ids += [idx] * 5
			cts += lbl_encoder.inverse_transform(np.argsort(y_pred[i])[::-1])[:5].tolist()


		sub = pd.DataFrame(np.column_stack((ids, cts)), columns=['id', 'country'])
		sub.to_csv(sub_path,index=False)

	if VALIDATE:
		print "Perform validation and save to {}".format(eval_path)
		
		y_eval = df_eval["country_destination"] 
		classes = lbl_encoder.classes_
		tmp = pd.DataFrame(y_pred, columns = ["pred_{}".format(i) for i in range(len(classes))])
		res_df = tmp.apply(lambda row: [x for (y,x) in sorted(zip(row, classes), reverse=True)], axis=1)
		res_df["true_destination"] = y_eval
		res_df["ndcg_5"] = res_df.apply(lambda row: score_predictions(np.reshape(row[:-1], (1,len(row)-1) ), 
	                                                   			 [row.true_destination], n_modes=5)[0], 
										axis=1 )
		print "Average evaluation score (ndcg): {}".format(res_df.ndcg_5.mean())
		res_df.to_csv(eval_path, index=False)




	
	
	