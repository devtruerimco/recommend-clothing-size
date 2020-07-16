# --------------
# import the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')


# Code starts here
df=pd.read_json(path,lines=True)

df.columns = df.columns.str.replace(' ','_')

missing_data=df.isnull().sum()

df.drop(columns=['waist', 'bust', 'user_name','review_text','review_summary','shoe_size','shoe_width'],inplace=True)

X=df.drop(columns=["fit"])

y=df["fit"]

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.33,random_state=6)




# Code ends here


# --------------
def plot_barh(df,col, cmap = None, stacked=False, norm = None):
    df.plot(kind='barh', colormap=cmap, stacked=stacked)
    fig = plt.gcf()
    fig.set_size_inches(24,12)
    plt.title("Category vs {}-feedback -  cloth {}".format(col, '(Normalized)' if norm else ''), fontsize= 20)
    plt.ylabel('Category', fontsize = 18)
    plot = plt.xlabel('Frequency', fontsize=18)


# Code starts here
g_by_category=df.groupby("category")

#g_by_category.value_counts()
cat_fit=g_by_category["fit"].value_counts()

cat_fit=cat_fit.unstack()


#plot_barh(cat_fit)
plot_barh(cat_fit,"category")

# Code ends here


# --------------
# Code starts here

#g_by_category.value_counts() by length
cat_len=g_by_category["length"].value_counts()

cat_len=cat_len.unstack()


#plot_barh(cat_len)
plot_barh(cat_len,"length")

# Code ends here


# --------------
# function to to convert feet to inches

def get_cms(x):
    if type(x) == type(1.0):
        return
    #print(x)
    try: 
        return (int(x[0])*30.48) + (int(x[4:-2])*2.54)
    except:
        return (int(x[0])*30.48)

# apply on train data    
X_train.height = X_train.height.apply(get_cms)

# apply on testing set
X_test.height = X_test.height.apply(get_cms)


# --------------
# Code starts here
#X_train.isnull().sum()

X_train=X_train.dropna(subset=["height","length","quality"])
X_test=X_test.dropna(subset=["height","length","quality"])

to_del_train=list(set(y_train.index)-set(X_train.index))
y_train.drop(to_del_train,inplace=True)

to_del_test=list(set(y_test.index)-set(X_test.index))
y_test.drop(to_del_test,inplace=True)

X_train["bra_size"].fillna((X_train["bra_size"].mean()),inplace=True)
X_test["bra_size"].fillna((X_test["bra_size"].mean()),inplace=True)

mode_1= X_train["cup_size"].mode()[0]
mode_2= X_test["cup_size"].mode()[0]

X_train["cup_size"]=X_train["cup_size"].replace(np.nan,mode_1)
X_test["cup_size"]=X_test["cup_size"].replace(np.nan,mode_2)

# Code ends here


# --------------
# Code starts here

X_train=pd.get_dummies(X_train,columns=["category","cup_size","length"])

X_test=pd.get_dummies(X_test,columns=["category","cup_size","length"])

# Code ends here


# --------------
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score


# Code starts here

#Fitting a decision tree model
model=DecisionTreeClassifier(random_state=6)

model.fit(X_train,y_train)

y_pred=model.predict(X_test)

#Checking accuracy and precision 
score=accuracy_score(y_test,y_pred)

precision=precision_score(y_test,y_pred,average=None)

print("accuracy score:",score)

print("precision score:",precision)




# Code ends here


# --------------
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier


# parameters for grid search
parameters = {'max_depth':[5,10],'criterion':['gini','entropy'],'min_samples_leaf':[0.5,1]}

# Code starts here
model=DecisionTreeClassifier(random_state=6)

grid=GridSearchCV(estimator=model,param_grid=parameters)

grid.fit(X_train,y_train)

y_pred=grid.predict(X_test)

accuracy=accuracy_score(y_test,y_pred)

print("accuracy score:",accuracy)

# Code ends here


