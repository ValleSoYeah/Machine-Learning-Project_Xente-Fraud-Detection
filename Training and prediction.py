import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import recall_score, confusion_matrix, matthews_corrcoef, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from Error_analysis import calculate_cost

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


RSEED= 42

df= pd.read_csv('data/training_preprocessed.csv')
print("File 'data/training_preprocessed.csv' was loaded")

y= df.FraudResult
df.drop("FraudResult", axis=1, inplace=True)
X= df



X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=RSEED)
print("Training data was split into Train and 25% Test")

X_test_unscaled = X_test.copy()
X_train.drop(["TransactionId"], axis=1, inplace=True)
X_test.drop(["TransactionId"], axis=1, inplace=True)

stdsc = StandardScaler()
X_train['ModAmount'] = stdsc.fit_transform(pd.DataFrame(X_train['ModAmount']))
X_test['ModAmount'] = stdsc.transform(pd.DataFrame(X_test['ModAmount']))
print("ModAmount column was standardized")


rf_best = RandomForestClassifier(max_features= 0.75, max_leaf_nodes = 80, min_samples_split = 5, n_estimators= 125, class_weight="balanced", random_state=RSEED)
rf_best.fit(X_train, y_train)
print("Model was fit to training data: RandomForestClassifier(max_features= 0.75, max_leaf_nodes = 80, min_samples_split = 5, n_estimators= 125, class_weight='balanced', random_state=RSEED)")

print("Testing model performance on 25% Test split")
y_pred_best = rf_best.predict(X_test)
print(confusion_matrix(y_test, y_pred_best))
print("----" * 10)
print(f"Recall is:{round(recall_score(y_test, y_pred_best),3)}")
print(f"F1 is:{round(f1_score(y_test, y_pred_best),3)}")
print(f"MCC is:{round(matthews_corrcoef(y_test, y_pred_best),3)}")
calculate_cost(X_test_unscaled, y_test, y_pred_best)


# Import test data for submission
X_sub = pd.read_csv('data/test_preprocessed.csv')
print("File 'data/test_preprocessed.csv' was loaded")
X_sub.drop(["TransactionId"], axis=1, inplace=True)

#create missing columns
X_sub["ProductId_12"] = 0
X_sub["other"] = 0

#remove excessive columns
X_cols= X_train.columns.to_list()
X_sub = X_sub[X_cols]

#scale ModAmount
X_sub['ModAmount'] = stdsc.transform(pd.DataFrame(X_sub['ModAmount']))
y_pred_sub = rf_best.predict(X_sub)
print("Predictions were made")

sample_submission = pd.read_csv('data/sample_submission.csv')
sample_submission["FraudResult"] = y_pred_sub
sample_submission.to_csv("data/sample_submission.csv", index=False)
print("Submission file was saved as 'data/sample_submission.csv'")