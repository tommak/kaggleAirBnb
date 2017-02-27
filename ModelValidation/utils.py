from __future__ import division

import pandas as pd
import numpy as np
import math
import xgboost as xgb
import matplotlib.pyplot as plt


from cross_validation import score_predictions
from scipy.stats import mannwhitneyu
from scipy.stats import ttest_ind

def get_levels_stat(df, excl_fields=None):
    res = pd.DataFrame([], columns=["field", "level", "count", "perc"])
    for field in df.columns:
        if field in excl_fields:
            continue
        counts = df[field].value_counts(dropna=False)
        perc = 100 * counts/df.shape[0]
        levels_stat = pd.DataFrame({ "field": field, "level" : perc.index, "perc" : perc, "count": counts })
        res = res.append(levels_stat, ignore_index=True)
    return res


def get_fixed_eval(ordred_targets, eval_res):
	eval_fixed = pd.DataFrame(np.vstack([ordred_targets]*eval_res.shape[0]), \
				columns=eval_res.columns[:len(ordred_targets)])
	eval_fixed["true_destination"] = eval_res["true_destination"]
	eval_fixed["ndcg_5"] = eval_fixed.apply(lambda row: score_predictions(np.reshape(row[:-1], (1,len(row)-1) ), 
	                                       [row.true_destination], n_modes=5)[0], axis=1 )
	return eval_fixed

def get_rnd_eval(path, eval_res):
	return load_or_calc_rndeval(path, eval_res)

def calc_rnd_eval(eval_res):
	s = []
	for i in range(eval_res.shape[0]):
		a = v[np.random.choice(v.index, 5, replace=False, p = v.values)]
		a.sort(ascending=False)
		s.append(list(a.index))
    
	rnd_eval = pd.DataFrame(s, columns=eval_res.columns[:5])
	rnd_eval["true_destination"] = eval_res["true_destination"]
	return rnd_eval.apply(lambda row: score_predictions(np.reshape(row[:-1], (1,len(row)-1) ), 
							[row.true_destination], n_modes=5)[0], axis=1 )



def load_or_calc_rndeval(path, eval_res):
	try:
		print "Load from {}".format(path)
		mean_rnd_df = pd.read_csv(path)
	except IOError:
		print "Start calculation"
		n = 100
		r = np.zeros(eval_res.shape[0])
		for i in range(n):
			if i > 0 and i % 5 == 0:
				print "{} runs calculated".format(i)
		r = r + calc_rnd_eval(eval_res).values
    
		mean_r = r/n
		mean_rnd_df = pd.DataFrame({"true_destination" : eval_res.true_destination, "mean_rnd_score": mean_r})
		mean_rnd_df.to_csv(path, index=False)

	return mean_rnd_df

def cohen_d(x,y):
        return (np.mean(x) - np.mean(y)) / math.sqrt((np.std(x, ddof=1) ** 2 + np.std(y, ddof=1) ** 2) / 2.0)

def perform_rank_test(ranks):
    res = pd.DataFrame([], columns = ["country_destination", "smpl1_size", "smpl2_size", \
                                      "stat", "pval",  "clez", "rb_corr"])
    for col in ranks.columns:
        if col == "true_destination":
            continue
        elif col == "NDF":
            smpl1 = ranks.loc[ranks.true_destination == col, col].values
            smpl2 = ranks.loc[(ranks.true_destination != col), col].values
            stat, pval = mannwhitneyu(smpl1, smpl2, alternative='less', use_continuity=False)

        else:
            smpl1 = ranks.loc[ranks.true_destination == col, col].values
            smpl2 = ranks.loc[(ranks.true_destination != col) & (ranks.true_destination != "NDF"), col].values
            stat, pval = mannwhitneyu(smpl1, smpl2, alternative='less', use_continuity=False)

        n1 = len(smpl1)
        n2 = len(smpl2)
        # Calculate Common language effect size
        k = 0
        for val1 in smpl1:
            for val2 in smpl2:
                if val1 < val2:
                    k = k + 1
        cles = float(k) / (n1*n2)
        
        # Calculate Rank-biserial correlation
        r = 1 - 2*stat/(n1*n2)
    
        res = res.append(pd.DataFrame([[col, n1, n2, stat, pval, cles, r,]], \
                                    columns = ["country_destination", "smpl1_size", "smpl2_size", \
                                               "stat", "pval", "clez", "rb_corr"]), ignore_index=True)
    return res


def perform_ttest(ranks):
    res = pd.DataFrame([], columns = ["country_destination", "smpl1_size", "smpl2_size", "avg_rank1", \
                                       "avg_rank2", "stat", "pval",  "cohen_d"])
    for col in ranks.columns:
        if col == "true_destination":
            continue
        elif col == "NDF":
            smpl1 = ranks.loc[ranks.true_destination == col, col].values
            smpl2 = ranks.loc[(ranks.true_destination != col), col].values
            stat, pval = ttest_ind(smpl1, smpl2)

        else:
            smpl1 = ranks.loc[ranks.true_destination == col, col].values
            smpl2 = ranks.loc[(ranks.true_destination != col) & (ranks.true_destination != "NDF"), col].values
            stat, pval = ttest_ind(smpl1, smpl2)

        n1 = len(smpl1)
        n2 = len(smpl2)
        # Calculate Cohen`s D
        d = cohen_d(smpl2, smpl1)
    
        res = res.append(pd.DataFrame([[col, n1, n2, np.mean(smpl1), np.mean(smpl2), stat, pval, d]], \
                                    columns = ["country_destination", "smpl1_size", "smpl2_size", "avg_rank1", \
                                               "avg_rank2", "stat", "pval", "cohen_d"]), ignore_index=True)
    return res


def plot_importance(fitted_clf, features, top):
	fig, [ax1,ax2] = plt.subplots(1,2, figsize=(12,5))

	xgb.plot_importance(fitted_clf, max_num_features=top, importance_type="weight",
                        title="Feature Importance \n(by number of times a feature appears in a tree\n",
                        ax=ax1)
	labels1 = [item.get_text() for item in ax1.get_yticklabels()]
	flabels1 = map(lambda v: features[int(v[1:])],labels1)
	ax1.set_yticklabels(flabels1)

	xgb.plot_importance(fitted_clf, max_num_features=top, importance_type="gain",
	                        title="Feature Importance \n(by average gain of splits which use the feature\n",
	                        ax=ax2)
	labels2 = [item.get_text() for item in ax2.get_yticklabels()]
	flabels2 = map(lambda v: features[int(v[1:])],labels2)
	ax2.set_yticklabels(flabels2)

	plt.subplots_adjust(wspace = 0.8)
	plt.show()
