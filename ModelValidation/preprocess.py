import pandas as pd
import numpy as np
from os.path import isfile
from sklearn.cross_validation import train_test_split


def read_users_ses(users_fname, sessions_fname, merger, merged_fname, actions_fname=None):
	"""
		Merge users and sessions data sets or read from existing file
	"""          
	try:
		df = pd.read_csv(merged_fname)
	except IOError:
		print "Merge with {} \n users from {} \n sessions from {}"\
					.format(merger.__name__, users_fname, sessions_fname)
		users = pd.read_csv(users_fname)
		sessions = pd.read_csv(sessions_fname)
		df = merger(users, sessions, users_ses_file=merged_fname, agg_actions_file=actions_fname)
		df.to_csv(merged_fname, index=False)
		print "Save into {}".format(merged_fname)
	return df

def split_users(users_fname, train_fname, test_fname, test_size=0.1, random_state=42):
	"""
		Split users data set into train and test parts or read from exsisting files
	"""
	if isfile(train_fname) and isfile(test_fname):
		train = pd.read_csv(train_fname)
		test = pd.read_csv(test_fname)
		return train, test
	else:
		print "Perform train test split" 
		users = pd.read_csv(users_fname)
		train, test = train_test_split(users, test_size=test_size, random_state=random_state)
		train.to_csv(train_fname, index=False)
		test.to_csv(test_fname, index=False)
		print "Save train into: {}, \nSave test into: {}".format(train_fname, test_fname)
	return train, test




def combine_action_info(row):
	""" Returns a string which combines variables action, action_type and action_detail
		from the specified row 
	"""
	res = ""
	if not pd.isnull(row.action):
		res = res + "|a|" + row.action
	if not pd.isnull(row.action_type):
		res = res + "|t|" + row.action_type
	if not pd.isnull(row.action_detail):
		res = res + "|d|" + row.action_detail
	return res

def aggregate_action_info(sessions):
 	"""
 	Create a list of all unique actions (unique triplets <action, action_type, action_detail>)
 	and combine it in aggregated action variable, add statistics on percentage of users. 
 	
 	Returns:
 		A data frame with unique triplets action, action_type, action_detail,
 		aggr_action - aggreagated action variable, which combines all three variable in one,
 		agg_action_code - coded action name with template "agg_action_N", so that  
 			agg_action_0 - is the most common action among others, agg_action_1 - the next most common
 			action after agg_action_0 and etc.
 		perc_users - percent of users, who performed that action
 	"""
	actions_info = sessions[["action", "action_type", "action_detail"]].drop_duplicates()
	actions_info.index=range(len(actions_info))
	actions_info["aggr_action"] = actions_info.apply(combine_action_info, axis=1)
	sessions_aggr = pd.merge(sessions, actions_info, on = ["action", "action_type", "action_detail"])
	agg_actions_info = sessions_aggr.groupby("aggr_action")["user_id"].agg({"num_users" : pd.Series.nunique}). \
	                                    reset_index().sort("num_users", ascending=False)
	agg_actions_info.index = range(len(agg_actions_info))
	agg_actions_info["agg_action_code"] = ["agg_action_%s" % i for i in range(len(agg_actions_info))]
	agg_actions_info["perc_users"] = agg_actions_info.num_users / sessions.user_id.nunique()
	agg_actions_info = pd.merge(actions_info, agg_actions_info, on = "aggr_action")

	return agg_actions_info

def add_action_grouping(agg_actions_info, key_words):
	""" 
	Specify a group of rarely used actions.
	Add additional grouping variable 'agg_action_code_gr' to the data frame 
	agg_actions_info. 
	Actions which were used by less than 1% of users are combined in group 'agg_action_other'.
	All actions from "other" group are further classified according to specific key words, if aggregated action 
	contains a key word than it is classified to the group 'agg_action_other_[key_word]'
	All other actions are stayed unchanged.
	"""

	agg_actions_info["agg_action_code_gr"] = agg_actions_info.agg_action_code
	agg_actions_info.loc[agg_actions_info.perc_users < 0.01, "agg_action_code_gr"] = "agg_action_other" 
	
	for key_word in key_words:
	    key_word_rows = agg_actions_info.aggr_action.str.contains(r'[\W_]+' + key_word)
	    agg_actions_info.loc[key_word_rows & (agg_actions_info.perc_users < 0.01), "agg_action_code_gr"] = "agg_action_other_" + key_word	

def users_ses_merge(users, sessions, test_users=None, \
								users_ses_file=None, test_users_ses_file=None,
								agg_actions_file=None):
	"""
	Merge useres with sessions by user_id. Session data set is preprocessed so that an action is 
	represented by one aggregated variable. Some grouping and name encoding of actions is performed.  
	Returns:
		users_ses (Data Frame) and test_users_ses (Data Frame) if required 
			full user record from input data frame users (test_users),
			list of all actions (agregated and grouped) with corresponding number of times it was performed by the user,
				(zero if not used by the user)
			duration - total duration of all sessions,
			num_actions - total number of actions performed,
			list of all devices with corresponding number of times it was used by the user 


	"""

	agg_actions_info = aggregate_action_info(sessions)
	
	key_words = ["host", "book", "wishlist", "reserv", "transl", "coupon"]
	add_action_grouping(agg_actions_info, key_words)

	sessions_aggr_coded = pd.merge(sessions, agg_actions_info[["action", "action_type", "action_detail", "agg_action_code_gr"]], \
							how="left", on = ["action", "action_type", "action_detail"])

	P1 = sessions_aggr_coded.pivot_table(index="user_id", columns="agg_action_code_gr", values="secs_elapsed", aggfunc=np.size)
	P2 = sessions_aggr_coded.groupby("user_id")["secs_elapsed"].agg({"duration" : np.sum, "num_actions" : np.size})
	P3 = sessions_aggr_coded.pivot_table(index="user_id", columns="device_type", values="secs_elapsed", aggfunc=np.size)

	sessions_pivot = pd.merge(pd.merge(P1, P2, left_index=True, right_index=True), P3,
	                          left_index=True, right_index=True)
							  
	users_ses = pd.merge(users, sessions_pivot, how="left", left_on = "id", right_index=True)
	
	if test_users:
		test_users_ses = pd.merge(test_users, sessions_pivot, how="left", left_on = "id", right_index=True)
		if test_users_ses_file:
			test_users_ses.to_csv(test_users_ses_file, index=False)

	if users_ses_file:
		users_ses.to_csv(users_ses_file, index=False)
	
	if agg_actions_file:
		agg_actions_info.to_csv(agg_actions_file, index=False)

	if test_users:
		return users_ses, test_users_ses
	else:
		return users_ses      

def get_sample_weight(class_weights, tr_labels):
	"""
		Return sample weights which correspond to required class weights
		Parameters
		-----------
		class_weights dict
			Dictionary with the following format: {class : required class weight}
		tr_labels pd.Series
	"""
	cl_weights = pd.Series(class_weights)
	real_class_weights = tr_labels.value_counts(normalize=True)
	m = pd.concat([cl_weights, real_class_weights], axis=1)
	new_sample_weights = m.iloc[:,0]/m.iloc[:,1] 
	new_sample_weights.name ="sample_weight"

	d = tr_labels.to_frame()
	d.columns = ["destination"]
	d = d.join(new_sample_weights, on="destination", how="left") 
	return d["sample_weight"].values


