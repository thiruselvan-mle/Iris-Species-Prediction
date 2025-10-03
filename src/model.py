import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import label_binarize

def split_x_y(df):
    X=df.drop('Species',axis=1)
    Y=df['Species']
    print('Features Columns:',X.columns)
    print('\n')
    print('Target Columns:','Species')
    return X, Y

def split_train_test(X,Y, test_size=0.2, random_state=42):
    X_train, X_test, Y_train, Y_test=train_test_split(X, Y, test_size=test_size, random_state=random_state)
    print("Training Sample:",X_train.shape[0])
    print("Testing Sample:",X_test.shape[0])
    return X_train, X_test, Y_train, Y_test

def algo_comparison(X_train, Y_train):
    models=[]
    models.append(('LR',LogisticRegression(max_iter=500)))
    models.append(('DT',DecisionTreeClassifier(random_state=42)))
    models.append(('RFC',RandomForestClassifier(random_state=42)))
    models.append(('KNN',KNeighborsClassifier()))
    models.append(('SVC',SVC()))
    models.append(('LDA',LinearDiscriminantAnalysis()))
    models.append(('NB',GaussianNB()))

    results=[]
    names=[]
    res=[]

    for name,model in models:
        Kfold=StratifiedKFold(n_splits=10, random_state=None)
        cv_results=cross_val_score(model, X_train, Y_train, cv=Kfold, scoring="accuracy")
        results.append(cv_results)
        names.append(name)
        res.append(cv_results.mean())
        print("%s:%f%%"%(name,cv_results.mean()*100))
    return results, names, res

def plot_algo_comparison(names, res):
    plt.figure(figsize=(10,5))
    plt.bar(names, res, color='g', width=0.5)
    plt.title('Algorithm Comparison')
    plt.xlabel('Names')
    plt.ylabel('Accuracy')
    plt.ylim(.920,1)
    plt.show()

def boxplot_algo_comparison(results, names):
    plt.figure(figsize=(10,5))
    plt.boxplot(results, tick_labels=names)
    plt.title('Algorithm Comparison')
    plt.xlabel('Names')
    plt.ylabel('Accuracy')
    plt.show()

def accu_table(names, res):
    results_df=({
    'Models': names,
    'Mean_Accuracy': [round(r*100,2) for r in res]
    })

    accu_table=pd.DataFrame(results_df).sort_values(by='Mean_Accuracy', ascending=False)
    print(accu_table)

def train_test(df, test_size=0.2, random_state=42):
    X=df.drop('Species',axis=1)
    Y=df['Species']

    X_train, X_test, Y_train, Y_test=train_test_split(X, Y, test_size=test_size, random_state=random_state)

    print("Training Sample:",X_train.shape)
    print("Testing Sample:",X_test.shape)
    return X_train, X_test, Y_train, Y_test

def train_model(X_train, Y_train, X_test):
    model=SVC(probability=True, random_state=42)
    model.fit(X_train, Y_train)

    y_pred=model.predict(X_test)
    y_score = model.decision_function(X_test)
    return y_pred, y_score, model

def model_accu(y_pred, Y_test):
    accu=accuracy_score(y_pred, Y_test)*100
    print(f'Accuracy_Score:{accu:.2f}')
    return accu

def plot_confusion_matrix(y_pred, Y_test):
    cm=confusion_matrix(y_pred,Y_test)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('SVC Confusion Matrix')
    plt.show()

def classification_rept(y_pred, Y_test):
    print(classification_report(y_pred, Y_test))

def roc_auc(Y_test, y_score):

    Y_test_bin = label_binarize(Y_test, classes=[0,1,2])


    fpr, tpr, roc_auc = {}, {}, {}
    for i in range(Y_test_bin.shape[1]):
        fpr[i], tpr[i], _ = roc_curve(Y_test_bin[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])


    plt.figure(figsize=(7,6))
    colors = ['orange', 'green', 'blue']
    for i, color in zip(range(Y_test_bin.shape[1]), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label=f"Class {i} (AUC = {roc_auc[i]:.2f})")

    plt.plot([0,1], [0,1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve - SVC (Decision Function)')
    plt.legend(loc="lower right")
    plt.show()
    return roc_auc

def accu_auc_table(roc_auc, accu):
    roc_auc_macro = np.mean(list(roc_auc.values()))

    accu_df = pd.DataFrame({
    'Metrics': ['Accuracy', 'AUC'],
    'Score': [round(accu, 2), round(roc_auc_macro, 2)]
    })

    display(accu_df)

def save_model(model, filename=r"D:\Thiru\ML_Projects\Iris-Species-Prediction\models\SVC.pkl"):
    joblib.dump(model,filename)
    print(f'model saved to {filename}')

def load_model(filename="D:\Thiru\ML_Projects\Iris-Species-Prediction\models\SVC.pkl"):
    model=joblib.load(filename)
    print(f'model loaded from {filename}')
    return model