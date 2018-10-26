#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""

Application - Using Python and Pandas to improve cross-selling and revenue generation in retail

Usage:
    application.py [<comma-separated-list-of-products]

"""

__author__ = "Andre Lessa"

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing
import sys

#
#
#

def create_datasets():
    # These data files can be downloaded from github.com/lessaworld/course

    orders = pd.read_excel('data/orders.xls')

    products = pd.read_excel('data/products.xls')

    order_products = orders.pivot_table(index='OrderNumber', columns='Product', 
                                        values='Quantity', fill_value=0)  
    
    return (orders, products, order_products)

#
#
#

def inspect_data():
    print("\n﹥﹥﹥ Inspecting the Orders Dataframe")

    print("\n﹥﹥﹥ orders.shape \n", orders.shape)

    print("\n﹥﹥﹥ orders.dtypes\n", orders.dtypes)

    print("\n﹥﹥﹥ orders.describe() \n", orders.describe())

    print("\n﹥﹥﹥ orders.head() \n", orders.head())

    print("\n﹥﹥﹥ orders.isnull().values.any() \n", orders.isnull().values.any())
    
    top_selling_products = orders.groupby(["Product"]).sum().\
                                sort_values("Quantity", ascending=False)

    print("\n﹥﹥﹥ top_selling_products.head \n", top_selling_products.head())

    print("\n\n﹥﹥﹥ Inspecting the Products Dataframe")
    
    print("\n﹥﹥﹥ products.head()\n", products.head())

    print("\n﹥﹥﹥ products.shape\n", products.shape)
  
    print("\n﹥﹥﹥ products.columns\n", products.columns)     
  
    products_leftmost_columns = products[products.columns[:-1]] 
    products_rightmost_column = products[products.columns[-1]] 
  
    print("\n﹥﹥﹥ products[products.columns[:-1]]\n", products_leftmost_columns.head()) 
  
    print("\n﹥﹥﹥ products[products.columns[-1]]\n", products_rightmost_column.head()) 

    print("\n﹥﹥﹥ products.isnull().values.any()\n", products.isnull().values.any())

    print("\n﹥﹥﹥ order_products # PIVOT TABLE\n", order_products)
    
    print("\n\n")
    
#
#
#

def visualize_quantity_histogram():
    sns.set_style('dark')  
    plt.figure(figsize=(8,6))  
    plt.rcParams['patch.force_edgecolor'] = True  
    orders['Quantity'].hist(bins=10)  
    plt.savefig("histogram.png")
    print("\n﹥﹥﹥ saved file: histogram.png\n")

#
#
#

def visualize_frequecy_heatmap():
    sns.set(style="white")
    sns.set(font_scale=0.8)

    same_dataset_correlation = order_products.corr()
    print("\n﹥﹥﹥ same_dataset_correlation\n", same_dataset_correlation)
    
    scaler = preprocessing.MinMaxScaler()
    scaled_correlation_array = scaler.fit_transform(same_dataset_correlation)
    
    scaled_correlation = pd.DataFrame(scaled_correlation_array, columns=list(same_dataset_correlation))
    scaled_correlation.index = same_dataset_correlation.index

    mask = np.zeros_like(scaled_correlation, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True

    f, ax = plt.subplots(figsize=(11, 9))

    sns.heatmap(scaled_correlation, mask=mask, cmap="Blues", center=0.5, square=True)
    
    plt.savefig("heatmap.png")                
    print("\n﹥﹥﹥ saved file: heatmap.png\n")

#
#
#

def find_best_correlations(product):
    product_record = products.loc[products['Product'] == product]
    product_category = product_record['Category'].values[0]

    product_scores = order_products[product]  
    similar_products_matrix = order_products.corrwith(product_scores)

    similar_products = pd.DataFrame(similar_products_matrix, columns=['Correlation'])  
    similar_products = similar_products.sort_values('Correlation', ascending=False).head(10) 
    similar_products = similar_products.join(products.set_index('Product'), on='Product')  

    similar_products = similar_products[(similar_products['InventoryLevel']!=0) & 
                                        (similar_products['Category']!= product_category)].\
                                        sort_values('Correlation', ascending=False)
    return similar_products

#
#
#

def find_recommendations(order):
    order_categories = []
    for product in order:
        product_record = products.loc[products['Product'] == product]
        order_categories.append(product_record['Category'].values[0])        

    similarity_scores = pd.DataFrame(columns=('Category', 'Product', 'Score'))
    idx = 0
    for product in order:
        correlations = find_best_correlations(product)
        for product_in_row, row in correlations.iterrows(): # return series_index, row
            if row['Category'] in order_categories or product_in_row in order:
                continue
            if row['Correlation'] > 0:
                similarity_scores.loc[idx] = [row['Category'], product_in_row, row['Correlation']]
                idx +=1

    recommendations = similarity_scores.groupby('Product').Score.sum().reset_index().\
                                                sort_values("Score", ascending=False)

    if len(recommendations) == 1:
        statement = "a {}".format(recommendations.iloc[0]["Product"])
    elif len(recommendations) > 1:
        if recommendations.iloc[0]["Score"] > 0.70:
            question = "a {}".format(recommendations.iloc[0]["Product"])
        else:
            statement = "a {} or a {}".format(recommendations.iloc[0]["Product"], 
                                              recommendations.iloc[1]["Product"])
    else:
        statement = "anything else"

    print("\n\"Can I offer you {} to go along with your order?\"\n\n".format(statement))


#
# 
#

if __name__ == "__main__":

    orders, products, order_products = create_datasets()
    
    if len(sys.argv) >= 2:
        order = [p.strip().capitalize() for p in (' '.join(sys.argv[1:])).split(",")]

        print("Checking recommendations for {} ".format(', '.join(order)))

        find_recommendations(order)

    else:
        inspect_data()

        visualize_quantity_histogram()

        visualize_frequecy_heatmap()
