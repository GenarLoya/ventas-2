import numpy as np
import pandas as pd
import mlxtend
from mlxtend.frequent_patterns import association_rules, apriori
from mlxtend.frequent_patterns import fpgrowth

df = pd.read_csv('ventas.csv')
print("------HEAD------")
print("Size of the dataframe: {}".format(df.shape))
print(df.head())

print("------PREPROCESSING------")
df_pivot = df.pivot_table(index='transaction_number',
                          columns='item', values='amount', aggfunc="sum").fillna(0).astype(int)

print("Size of the pivot table: {}".format(df_pivot.shape))
print(df_pivot)

print("------APRIORI------")


def encode(x):
    if x <= 0:
        return 0
    else:
        return 1


df_encoded = df_pivot.map(encode)

support = 0.01
frequent_items = apriori(df_encoded, min_support=support, use_colnames=True)
print(frequent_items)

frequent_items.sort_values(by='support', ascending=True)

metric = 'lift'
min_treshold = 1

rules = association_rules(frequent_items, metric=metric,
                          min_threshold=min_treshold)
rules.reset_index(drop=True).sort_values(
    by='confidence', ascending=False, inplace=True)
print(rules)
