from nose.tools import *
from ModelValidation.cross_validation import  *

def setup():
    print "SETUP!"

def teardown():
    print "TEAR DOWN!"

def  dcg_at_k_test():
	r = [3, 2, 3, 0, 0, 1, 2, 2, 3, 0]
	assert dcg_at_k(r, 1) == 3
	assert dcg_at_k(r, 1, method=1) ==  3.0
	assert dcg_at_k(r, 2) ==  5.0
	assert abs(dcg_at_k(r, 2, method=1) - 4.2618) < 0.0001
	assert abs(dcg_at_k(r, 10) - 9.60512) < 0.0001
	assert abs(dcg_at_k(r, 11) - 9.6051) < 0.0001

	
def ndcg_at_k_test():
	r = [3, 2, 3, 0, 0, 1, 2, 2, 3, 0]
	assert ndcg_at_k(r, 1) ==  1.0
	r = [2, 1, 2, 0]
	assert abs(ndcg_at_k(r, 4) - 0.9203) < 0.0001
	assert abs(ndcg_at_k(r, 4, method=1) - 0.9652) < 0.0001
	assert ndcg_at_k([0], 1) ==  0.0
	assert ndcg_at_k([1], 2) ==  1.0
	
def score_predictions_test():
	preds = [['US','FR'],['FR','US'],['FR','US']]
	truth = ['US','US','FR']
	scores = score_predictions(preds, truth)
	assert len(scores) == 3
	assert scores[0] == 1.0 
	assert abs(scores[1] - 0.6309) < 0.0001 
	assert scores[2] == 1.0
	
