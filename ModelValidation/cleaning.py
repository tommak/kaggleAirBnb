import numpy as np
import pandas as pd

def treat_missing_values(df):
	df_new = df.fillna(-1)
	return df_new

def prepare_data(df, num_features,  cat_features, target=None):
	""" Transform data into format suitable for sklearn algorithms,
		add dummy variables for categorical features

	Parameters
    ----------
    df : pd.DataFrame
    	Input data frame
    num_features: List of numerical feature names
	cat_features : List of categorical feature names

	Returns
	---------
	X : ndarray
		A two dimensional array containing data for specified numerical features and 
		categorical features coded with dummy variables
	"""

	algo_df = pd.DataFrame()
	
	for feature in num_features:
		algo_df[feature] = df[feature]
	
	for f in cat_features:
		df_dummy = pd.get_dummies(df[f], prefix=f)
		algo_df = pd.concat((algo_df, df_dummy), axis=1)

	return algo_df


def clean_manage_users_ses(df):
	
	
	#####Feature engineering#######
	#date_account_created
	dac = np.vstack(df.date_account_created.astype(str).apply(lambda x: list(map(int, x.split('-')))).values)
	df['dac_year'] = dac[:,0]
	df['dac_month'] = dac[:,1]
	df['dac_day'] = dac[:,2]
	
	#timestamp_first_active
	tfa = np.vstack(df.timestamp_first_active.astype(str).apply(lambda x: list(map(int, [x[:4],x[4:6],x[6:8],x[8:10],x[10:12],x[12:14]]))).values)
	df['tfa_year'] = tfa[:,0]
	df['tfa_month'] = tfa[:,1]
	df['tfa_day'] = tfa[:,2]
	
	#Age
	av = df.age.values
	df['age'] = np.where(np.logical_or(av<14, av>95), -1, av)
	
	#Add duration_days
	df["duration_days"] = df.duration / (3600.0 * 24 )

	#Fill zeros for actions
	action_cols = df.columns[df.columns.str.contains('agg_action')]
	df_cl = df
	df_cl.loc[df_cl.num_actions>0, action_cols] = df[df.num_actions>0][action_cols].fillna(0)
	for col in action_cols:
		df_cl[col] = df_cl[col]/df_cl.num_actions 
	
	
	#Fill other na with -1
	df_cl =  treat_missing_values(df_cl)

	
	cat_features = ['gender', 'signup_method', 'signup_flow', 'language', 'affiliate_channel', 'affiliate_provider', 'first_affiliate_tracked', 'signup_app', 'first_device_type', 'first_browser']
	num_features = ['age', 'dac_year', 'dac_month', 'dac_day', 'tfa_year', 'tfa_month', 'tfa_day', 'duration_days']
	num_features.extend(action_cols)
		
	return  df_cl, num_features, cat_features

def clean_manage_users_ses_ind(df):
	"""
	Prepare data for classification. Action variables are presented in 1 and 0 format indicating whether specific action was made by user.
	"""
	
	df_cl, num_features, cat_features = clean_manage_users_ses(df)
	action_cols = df_cl.columns[df_cl.columns.str.contains('agg_action')]
	for col in action_cols:
		df_cl[col] = np.sign(df_cl[col])
	return df_cl, num_features, cat_features

def clean_manage_users_ses_FE(df):
	"""
	Prepare data for classification. Action variables are presented in 1 and 0 format indicating whether specific action was made by user.
	"""
	
	df_cl, num_features, cat_features = clean_manage_users_ses_ext(df)
	
	sessions = pd.read_csv("..\data\\sessions.csv")
	df_cl["PrimeDevice"] = df_cl[sessions.device_type.unique()].idxmax(axis=1)
	vc = df_cl[df_cl.PrimeDevice.notnull()]["PrimeDevice"].value_counts(normalize=True)
	comb_values = vc[vc<0.01].index
	df_cl["PrimeDevice_rd"]=df_cl.PrimeDevice.apply(lambda x: 'Other' if x in comb_values else x)

	
	df_cl["num_devices"] = df_cl[sessions.device_type.unique()].notnull().sum(axis=1)
	df_cl["num_devices"] = df_cl["num_devices"].replace(0, np.nan) 
	
	vc = df_cl.signup_flow.value_counts(normalize=True)
	comb_values = vc[vc<0.01].index	
	df_cl["signup_flow_rd"]=df_cl.signup_flow.apply(lambda x: 'Other' if x in comb_values else x)
	df_cl["signup_method_rd"] = df_cl.signup_method
	df_cl.signup_method_rd[(df_cl.signup_method_rd=="facebook") | (df_cl.signup_method_rd=="google")] = "with_account"

	def device_type(device):
		new_type = device
		if device in ["SmartPhone (Other)"]:
			new_type = "Other/Unknown"
		return new_type

	df_cl["first_device_type_rd"] = df_cl.first_device_type.apply(device_type)
	vc = df_cl.first_browser.value_counts(normalize=True, dropna=False)
	incl_values = vc[vc>0.005].index
	df_cl["first_browser_rd"] = df_cl.first_browser.apply(lambda x: x if x in incl_values else 'Other')
	
	df_cl["gender_rd"] = df_cl.gender.apply(lambda x: "-unknown-" if x=="OTHER" else x)
	df_cl["language_rd"] = df_cl.language.apply(lambda x: "en" if x=="en" else "non-en" )
	
	df_cl["first_affiliate_tracked_rd"] = df_cl.first_affiliate_tracked.apply(lambda x: "tracked-other" if x=="local ops" else x)
	def aff_provider_gr(pr):
		gr = pr
		if pr in ["baidu", "naver", "daum", "yandex"]:
			gr = "non-en search"
		if pr in ["facebook", "facebook-open-graph", "meetup", "wayn"]:
			gr = "social net"
		if pr in ["bing", "vast", "yahoo"]:
			gr = "search eng"
		return gr

	df_cl["affiliate_provider_rd"] = df_cl.affiliate_provider.apply(aff_provider_gr)

	cat_features = ['gender_rd', 'signup_method_rd', 'signup_flow_rd', 'language_rd', 'affiliate_channel', 'affiliate_provider_rd', 'first_affiliate_tracked_rd', 
					'signup_app', 'first_device_type_rd', 'first_browser_rd', "PrimeDevice_rd"]
	num_features.append('num_devices')
	
	
	return df_cl, num_features, cat_features


def clean_manage_users_ses_FE_2(df):
	"""
	Prepare data for classification. Action variables are presented in 1 and 0 format indicating whether specific action was made by user.
	"""
	
	df_cl, num_features, cat_features = clean_manage_users_ses_ext(df)
	
	vc = df_cl.signup_flow.value_counts(normalize=True)
	comb_values = vc[vc<0.01].index	
	df_cl["signup_flow_rd"]=df_cl.signup_flow.apply(lambda x: 'Other' if x in comb_values else x)
	df_cl["signup_method_rd"] = df_cl.signup_method
	df_cl.signup_method_rd[(df_cl.signup_method_rd=="facebook") | (df_cl.signup_method_rd=="google")] = "with_account"

	def device_type(device):
		new_type = device
		if device in ["SmartPhone (Other)"]:
			new_type = "Other/Unknown"
		return new_type

	df_cl["first_device_type_rd"] = df_cl.first_device_type.apply(device_type)
	vc = df_cl.first_browser.value_counts(normalize=True, dropna=False)
	incl_values = vc[vc>0.005].index
	df_cl["first_browser_rd"] = df_cl.first_browser.apply(lambda x: x if x in incl_values else 'Other')
	
	df_cl["gender_rd"] = df_cl.gender.apply(lambda x: "-unknown-" if x=="OTHER" else x)
	df_cl["language_rd"] = df_cl.language.apply(lambda x: "en" if x=="en" else "non-en" )
	
	df_cl["first_affiliate_tracked_rd"] = df_cl.first_affiliate_tracked.apply(lambda x: "tracked-other" if x=="local ops" else x)
	def aff_provider_gr(pr):
		gr = pr
		if pr in ["baidu", "naver", "daum", "yandex"]:
			gr = "non-en search"
		if pr in ["facebook", "facebook-open-graph", "meetup", "wayn"]:
			gr = "social net"
		if pr in ["bing", "vast", "yahoo"]:
			gr = "search eng"
		return gr

	df_cl["affiliate_provider_rd"] = df_cl.affiliate_provider.apply(aff_provider_gr)

	cat_features = ['gender_rd', 'signup_method_rd', 'signup_flow_rd', 'language_rd', 'affiliate_channel', 'affiliate_provider_rd', 'first_affiliate_tracked_rd', 
					'signup_app', 'first_device_type_rd', 'first_browser_rd', "PrimeDevice_rd"]
	num_features.append('num_devices')
	
	
	return df_cl, num_features, cat_features

	
	
def clean_manage_users_ses_FE_1(df):
	"""
	Prepare data for classification. Action variables are presented in 1 and 0 format indicating whether specific action was made by user.
	"""
	
	df_cl, num_features, cat_features = clean_manage_users_ses_ext(df)
	
	sessions = pd.read_csv("..\data\\sessions.csv")
	df_cl["PrimeDevice"] = df_cl[sessions.device_type.unique()].idxmax(axis=1)
	vc = df_cl[df_cl.PrimeDevice.notnull()]["PrimeDevice"].value_counts(normalize=True)
	comb_values = vc[vc<0.01].index
	df_cl["PrimeDevice_rd"]=df_cl.PrimeDevice.apply(lambda x: 'Other' if x in comb_values else x)

	
	df_cl["num_devices"] = df_cl[sessions.device_type.unique()].notnull().sum(axis=1)
	df_cl["num_devices"] = df_cl["num_devices"].replace(0, np.nan) 
	
	cat_features.append("PrimeDevice_rd")
	#num_features.append('num_devices')
	
	
	return df_cl, num_features, cat_features
	
	
	
def clean_manage_users_ses_ext(df):
	"""
	Prepare data for classification. Add num_actions variable to analysis
	"""
	
	df_cl, num_features, cat_features = clean_manage_users_ses_ind(df)
	num_features.append("num_actions")
	return df_cl, num_features, cat_features
	

def clean_manage_kaggle(df):
	
	df = df.fillna(-1)

	#####Feature engineering#######
	#date_account_created
	dac = np.vstack(df.date_account_created.astype(str).apply(lambda x: list(map(int, x.split('-')))).values)
	df['dac_year'] = dac[:,0]
	df['dac_month'] = dac[:,1]
	df['dac_day'] = dac[:,2]
	
	#timestamp_first_active
	tfa = np.vstack(df.timestamp_first_active.astype(str).apply(lambda x: list(map(int, [x[:4],x[4:6],x[6:8],x[8:10],x[10:12],x[12:14]]))).values)
	df['tfa_year'] = tfa[:,0]
	df['tfa_month'] = tfa[:,1]
	df['tfa_day'] = tfa[:,2]
	
	#Age
	av = df.age.values
	df['age'] = np.where(np.logical_or(av<14, av>100), -1, av)

	cat_features = ['gender', 'signup_method', 'signup_flow', 'language', 'affiliate_channel', 'affiliate_provider', 'first_affiliate_tracked', 'signup_app', 'first_device_type', 'first_browser']
	num_features = ['age', 'dac_year', 'dac_month', 'dac_day', 'tfa_year', 'tfa_month', 'tfa_day']
	return  df, num_features, cat_features
	
	
def clean_manage_kaggle_v1(df):
	cl_df, num_features, cat_features = clean_manage_kaggle(df)
	cl_df["date_account_created"] = pd.to_datetime(df["date_account_created"])
	cl_df['dac_wday'] = cl_df["date_account_created"].map(lambda d: d.weekday())
	num_features.append("dac_wday")
	return cl_df, num_features, cat_features
	


def clean_manage_v2(df):
	
	cl_df, num_features, cat_features = clean_manage_v1(df)
	
	dac = np.vstack(cl_df.date_account_created.astype(str).apply(lambda x: list(map(int, x.split('-')))).values)
	cl_df['dac_year'] = dac[:,0]
	cl_df['dac_month'] = dac[:,1]
	cl_df['dac_day'] = dac[:,2]
	
	#timestamp_first_active
	tfa = np.vstack(cl_df.timestamp_first_active.astype(str).apply(lambda x: list(map(int, [x[:4],x[4:6],x[6:8],x[8:10],x[10:12],x[12:14]]))).values)
	cl_df['tfa_year'] = tfa[:,0]
	cl_df['tfa_month'] = tfa[:,1]
	cl_df['tfa_day'] = tfa[:,2]

	cl_df['dac_wday'] = cl_df["date_account_created"].map(lambda d: d.weekday())
	
	num_features.extend(['dac_year', 'dac_month', 'dac_day', 'tfa_year', 'tfa_month', 'tfa_day'])
	cat_features.append("dac_wday")
	
	return cl_df, num_features, cat_features
	
def clean_manage_v3(df):
	
	cl_df, num_features, cat_features = clean_manage_v1(df)
	
	dac = np.vstack(cl_df.date_account_created.astype(str).apply(lambda x: list(map(int, x.split('-')))).values)
	cl_df['dac_year'] = dac[:,0]
	cl_df['dac_month'] = dac[:,1]
	cl_df['dac_day'] = dac[:,2]
	
	#timestamp_first_active
	tfa = np.vstack(cl_df.timestamp_first_active.astype(str).apply(lambda x: list(map(int, [x[:4],x[4:6],x[6:8],x[8:10],x[10:12],x[12:14]]))).values)
	cl_df['tfa_year'] = tfa[:,0]
	cl_df['tfa_month'] = tfa[:,1]
	cl_df['tfa_day'] = tfa[:,2]

	cl_df['dac_wday'] = cl_df["date_account_created"].map(lambda d: d.weekday())
	
	num_features.extend(['dac_year', 'dac_month', 'dac_day', 'tfa_year', 'tfa_month', 'tfa_day', 'dac_wday'])
	
	cl_df = cl_df.fillna(-1)
	
	return cl_df, num_features, cat_features


	

def clean_manage_v1(df):
	
	
	
	cl_df = df.copy()
	cl_df["date_account_created"] = pd.to_datetime(df["date_account_created"])
	cl_df["first_active"] = pd.to_datetime(df["timestamp_first_active"], format='%Y%m%d%H%M%S')
	#cl_df["gender"] = cl_df["gender"].replace("-unknown-", np.nan)
	#cl_df["first_browser"] = df["first_browser"].replace("-unknown-", np.nan)
	#cl_df["first_device_type"] = df["first_device_type"].replace("Other/Unknown", np.nan)
	cl_df["age"] = df["age"].map(lambda age: age if (age>=18 and age<=100) else None)
	
	def device_type_3(device):
		dtype = None
		if device in ["Android Phone", "SmartPhone (Other)", "iPhone"]:
			dtype = "Phone"
		if device in ["Android Tablet", "iPad"]:
			dtype = "Tablet"
		if device in ["Desktop (Other)", "Mac Desktop", "Windows Desktop"]:
			dtype = "Desktop"
		return dtype

	#cl_df["lan_group"] = cl_df["language"].map(lambda l: "en" if l=="en" else "non-en")
	cl_df["device_type"] = cl_df["first_device_type"].map(device_type_3)

	def aff_provider_gr(pr):
		gr = pr
		if pr in ["baidu", "naver", "daum", "yandex"]:
			gr = "non-en search"
		if pr in ["facebook", "facebook-open-graph", "meetup", "wayn"]:
			gr = "social net"
		if pr in ["bing", "vast", "yahoo"]:
			gr = "search eng"
		return gr
	
	cl_df["aff_provider"] = cl_df["affiliate_provider"].map(aff_provider_gr)
	cl_df["signup_month"] = pd.Series([d.month for d in cl_df["date_account_created"]])
	
	def signup_flow_gr(fl):
		gr = fl
		if fl in [1, 5, 23, 25]:
			gr = "Gr1"
		if fl in [12, 20, 24]:
			gr = "Gr2"
		if fl in [2, 3, 6]:
			gr = "Gr3"
		if fl == 0:
			gr = "Gr4"
		if fl in [4, 8, 10, 15, 16, 21]:
			gr = "Gr5"
		return gr

	cl_df["signup_flow_gr"] = cl_df["signup_flow"].map(signup_flow_gr)
	
	#cl_df = cl_df.fillna(-1)

	cat_features = ['gender', 'signup_method', 'signup_flow_gr', 'language', 'affiliate_channel', 'aff_provider', 
				'first_affiliate_tracked', 'signup_app', 'device_type', 'first_browser', 'signup_month']
	
	#cat_features = ['device_type', 'lan_group', 'aff_provider', 'first_affiliate_tracked', 'affiliate_channel', 'signup_month', 'signup_flow_gr']
	num_features = ['age']
	return  cl_df, num_features, cat_features