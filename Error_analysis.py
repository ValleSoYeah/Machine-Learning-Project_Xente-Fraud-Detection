import pandas as pd
import numpy as np

def calculate_cost(X_test_unscaled, y_test, y_pred):
    '''Calculates the money saved by the model, returns the combined dataframe and the money saved by the model'''
    test_comb = X_test_unscaled.copy()
    test_comb["y_true"] = y_test
    test_comb["y_pred"] = y_pred
    false_positive = test_comb[test_comb.y_true < test_comb.y_pred]
    false_negative = test_comb[test_comb.y_true > test_comb.y_pred]
    correct = test_comb[test_comb.y_true == test_comb.y_pred]
    true_positive = correct[correct.y_pred == 1]
    true_negative = correct[correct.y_pred == 0]
    reimbursements = false_negative[false_negative.SignAmount == 1].ModAmount.sum()
    avoided_reimbursements = true_positive[true_positive.SignAmount == 1].ModAmount.sum()
    print(f"You detected {true_positive.shape[0]} fraudulent transactions")
    print(f"You saved the company {round((avoided_reimbursements)/1e6, 2)} Million UGX")
    print(f"You missed {false_negative.shape[0]} fraudulent transactions")
    print(f"The company now has to reimburse frauds with a total of {round((reimbursements)/1e6, 2)} Million UGX")
    print(f"Total money saved is {round((avoided_reimbursements - reimbursements)/1e6, 2)} Million UGX")
    print(f"You incorrectly flagged {false_positive.shape[0]} legit transactions as fraudulent")
    #return test_comb, (avoided_reimbursements - reimbursements)

def get_classification_dfs(X_test_unscaled, y_test, y_pred):
    test_comb = X_test_unscaled.copy()
    test_comb["y_true"] = y_test
    test_comb["y_pred"] = y_pred
    false_positive = test_comb[test_comb.y_true < test_comb.y_pred]
    false_negative = test_comb[test_comb.y_true > test_comb.y_pred]
    correct = test_comb[test_comb.y_true == test_comb.y_pred]
    true_positive = correct[correct.y_pred == 1]
    true_negative = correct[correct.y_pred == 0]
    return true_negative, false_positive, false_negative, true_positive