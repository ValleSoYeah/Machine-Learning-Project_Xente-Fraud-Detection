import pandas as pd
import numpy as np
import warnings
import datetime

def preproc(df):
    # Transform type object to type datetime
    df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'])
    df['TransactionStartTime'].head()

    # Split date and time
    df['Date'] = df['TransactionStartTime'].dt.date
    df['Time'] = df['TransactionStartTime'].dt.time

    # Split Date into year, month, week and day
    df['Year'] = df['TransactionStartTime'].dt.year
    df['Month'] = df['TransactionStartTime'].dt.month
    df['Week'] = df['TransactionStartTime'].dt.isocalendar().week
    df['Day'] = df['TransactionStartTime'].dt.day



    # 0 = Monday - 6 = Sunday
    df['Day_of_week'] = df['TransactionStartTime'].dt.dayofweek
    df['is_workday'] = df['Day_of_week'].apply(lambda x : True if x != 5 and x != 6 else False).astype(int)

    df["Time"] = df["Time"].astype(str)
    df['is_worktime'] = df['Time'].apply(lambda x : True if x >= '07:00:00' and x <= '18:00:00' else False).astype(int)
    df.drop("Time", axis=1, inplace=True)

    # drop redundant columns
    df.drop("Date", axis=1, inplace=True)
    df.drop("TransactionStartTime", axis=1, inplace=True)

    df["TransactionId"]=df.TransactionId.str.split('_').str[-1]
    df["BatchId"]=df.BatchId.str.split('_').str[-1]
    df["AccountId"]=df.AccountId.str.split('_').str[-1]
    df["SubscriptionId"]=df.SubscriptionId.str.split('_').str[-1]
    df["CustomerId"]=df.CustomerId.str.split('_').str[-1]

    df.drop("CountryCode", axis=1, inplace=True)
    df.drop("CurrencyCode", axis=1, inplace=True)
    df.drop("Value", axis=1, inplace=True)

    # create new column SignAmount with the sign of Amount (0 for - and 1 for +)
    df["SignAmount"] = df["Amount"].apply(lambda x: 1 if x >= 0 else 0)

    # create new column AbsAmount with the absolute of Amount
    df.eval("ModAmount = abs(Amount)", inplace=True)

    # delete Amount column
    df.drop("Amount", axis=1, inplace=True)

    # create dummy variable for ProviderId, ProductId, ProductCategory, ChannelId and PricingStrategy
    # join the 5 new dataframes with df

    df = df.join(pd.get_dummies(df["ProviderId"]))
    df = df.join(pd.get_dummies(df["ProductId"]))
    df = df.join(pd.get_dummies(df["ProductCategory"]))
    df = df.join(pd.get_dummies(df["ChannelId"]))

    # add string "PricingStrategy_" to all entries in column "PricingStrategy"
    df["PricingStrategy"] ='PricingStrategy_' + df["PricingStrategy"].astype(str)
    df = df.join(pd.get_dummies(df["PricingStrategy"]))
    

    df.drop(["ProviderId","ProductId","ProductCategory","ChannelId","PricingStrategy"],axis=1,inplace=True)

    df["batch_size"]= df.groupby(["BatchId"]).transform("count")["TransactionId"]
    df.drop("BatchId", axis=1, inplace=True)

    #create new feature to see how many transactions are on record for the customerID of this transaction
    df["total_transactions_by_customer"] = df.groupby(["CustomerId"]).transform("count")["TransactionId"]

    #create new feature to see how many transactions were made this month by the customerID of this transaction
    df["transactions_by_customer_this_month"] = df.groupby(["CustomerId", "Year", "Month"]).transform("count")["TransactionId"]

    #create new feature to see how many transactions were made this week by the customerID of this transaction
    df["transactions_by_customer_this_week"] = df.groupby(["CustomerId", "Year", "Week"]).transform("count")["TransactionId"]

    #create new feature to see how many transactions were made this week by the customerID of this transaction
    df["transactions_by_customer_this_day"] = df.groupby(["CustomerId", "Year", "Month", "Day"]).transform("count")["TransactionId"]

    #Unusual amounts of transaction in current timeframe?
    df["day_vs_week"] = df["transactions_by_customer_this_day"] / df["transactions_by_customer_this_week"]
    df["day_vs_month"] = df["transactions_by_customer_this_day"] / df["transactions_by_customer_this_month"]
    df["week_vs_month"] = df["transactions_by_customer_this_week"] / df["transactions_by_customer_this_month"]

    df.drop(["AccountId","SubscriptionId","CustomerId", "Day_of_week"], axis=1, inplace=True)

    df.set_index("TransactionId")
    return df

df_train = pd.read_csv("data/training.csv")
df_train = preproc(df_train)
df_train.to_csv("data/training_preprocessed.csv", index=False)


df_test = pd.read_csv("data/test.csv")
df_test = preproc(df_test)
df_test.to_csv("data/test_preprocessed.csv", index=False)
