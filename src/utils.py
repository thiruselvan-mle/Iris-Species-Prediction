import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def plot_feature_distribution(df):
    df.hist(figsize=(10,8), bins=15, edgecolor='black')
    plt.suptitle("Feature Distributions")
    plt.show()

def pairplot_features(df):
    sns.pairplot(df, hue='Species', palette='Set2')
    plt.show()

def plot_correlation(df):
    corr = df.corr()
    plt.figure(figsize=(8,6))
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title("Feature Correlation Matrix")
    plt.show()

def plot_species_distribution(df):
    sns.countplot(data=df, x='Species', hue=None, color='skyblue')
    plt.title("Species Distribution")
    plt.show()