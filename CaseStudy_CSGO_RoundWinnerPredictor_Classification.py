#!/usr/bin/env python
# coding: utf-8

# In[1]:


import warnings
warnings.filterwarnings('ignore')
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


pd.set_option('display.max_columns',None)
pd.set_option('display.max_rows', None)
np.set_printoptions(suppress=True)


# In[3]:


df=pd.read_csv("E:/Study/ML tuts/Case Studies/CSGO/csgo_round_snapshots.csv")
df.head()


# In[4]:


df.info()


# In[5]:


df.shape


# In[6]:


df.nunique()


# In[7]:


df['bomb_planted'] = df['bomb_planted'].astype(str)


# In[8]:


df['bomb_planted'].unique()


# In[9]:


df_cat=df.select_dtypes(object,bool)
df_num=df.select_dtypes(['int64','float64'])


# In[10]:


df_num.head()


# In[11]:


df_cat.head()


# In[12]:


from sklearn.preprocessing import LabelEncoder


# In[13]:


for col in df_cat:
    le=LabelEncoder()
    df_cat[col]=le.fit_transform(df_cat[col])
df_cat.head()


# In[14]:


df_cat.nunique()


# In[15]:


new_df=pd.concat([df_num,df_cat],axis=1)
new_df.head()


# In[16]:


new_df.info()


# In[17]:


new_df['round_winner'].value_counts()


# In[18]:


sns.countplot(new_df['round_winner'])
plt.show()


# 1  - Terrorist         - 62406
# 
# 0  - Counter-Terrorist - 60004

# In[19]:


sns.countplot(new_df['round_winner'],hue=new_df['map'])
plt.show()


# # Implementing Logistic Regression

# In[20]:


from sklearn.model_selection import train_test_split


# In[21]:


x=new_df.drop(['round_winner','ct_score','t_score'],axis=1)
y=new_df['round_winner']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3)


# In[22]:


from sklearn.linear_model import LogisticRegression


# In[23]:


logr=LogisticRegression()
logr.fit(x_train,y_train)
y_pred=logr.predict(x_test)
print(y_pred)


# In[24]:


np.set_printoptions(threshold=sys.maxsize)


# In[25]:


print(y_pred)


# In[26]:


y_predicted=logr.predict_proba(x_test)
print(y_predicted)


# In[27]:


print("Intercept : ",logr.intercept_)
print("Slope : ",logr.coef_)


# In[28]:


logr.score(x_test,y_test)


# # Confusion Matrix
# -------------------------------

# In[29]:


from sklearn.metrics import confusion_matrix


# In[30]:


confusion_matrix(y_test,y_pred)


# In[31]:


sns.countplot(y_test)
plt.show()


# In[32]:


tn,fp,fn,tp=confusion_matrix(y_test,y_pred).ravel()


# In[33]:


print("Confusion Matrix : ")
print("----------------------------------------------")
print("TP=",tp,"FP=",fp)
print("FN=",fn," TN=",tn)


# ![image.png](attachment:image.png)

# In[34]:


from sklearn.metrics import accuracy_score,recall_score,precision_score,f1_score


# In[35]:


accuracy_score(y_test,y_pred)


# In[36]:


recall_score(y_test,y_pred)


# In[37]:


precision_score(y_test,y_pred)


# In[38]:


f1_score(y_test,y_pred)


# # ROC AUC
# -------------------------------

# In[39]:


from sklearn.metrics import roc_auc_score


# In[40]:


print(roc_auc_score(y_test,y_pred))
#roc_curve is written so that ROC point corresponding to 
#the highest threshold (fpr[0], tpr[0]) is always (0, 0). 
#If this is not the case, a new threshold is created with an arbitrary value of max(y_score)+1


# In[41]:


from sklearn.metrics import roc_curve


# In[42]:


fpr,tpr,threshold=roc_curve(y_test,y_pred)


# In[43]:


fpr


# In[44]:


tpr


# In[45]:


threshold


# In[46]:


plt.plot(fpr,tpr,'r-')
plt.plot([0,1],[0,1],'k-',label="50% correct")  # first [] indicates x coordinates and the next [] are y coordinates
plt.plot([0,0,1],[0,1,1],'g-',label="excellent")
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.show()


# # Implementing ANOVA Feature Selection

# In[47]:


from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression


# In[48]:


anova=SelectKBest(score_func=f_regression,k=54)
anova.fit(x_train,y_train)
x_train_anova=anova.transform(x_train)
x_test_anova=anova.transform(x_test)


# In[49]:


scores_df=pd.DataFrame(anova.scores_)
columns_df=pd.DataFrame(x.columns)
featureScore_anova=pd.concat([columns_df,scores_df],axis=1)
featureScore_anova.columns=['Feature_Names','Score']
featureScore_anova


# In[50]:


featureScore_anova.sort_values("Score",ascending=False)


# In[51]:


featureScore_anova.nlargest(54, "Score")


# In[52]:


logr.fit(x_train_anova,y_train)


# In[53]:


print("Bias of Logistic Regression after Annova test = ",logr.score(x_train_anova,y_train))
print("Variance of Logistic Regression after Annova test =",logr.score(x_test_anova,y_test))


# In[54]:


y_predicted=logr.predict(x_test_anova)


# In[55]:


tbl=pd.DataFrame(list(zip(y_test,y_predicted)),columns=['y_Observed','y_predicted'])
print(tbl)


# # Implementing Decision Tree

# In[56]:


from sklearn.tree import DecisionTreeClassifier


# In[57]:


from sklearn import tree


# In[58]:


fig=plt.gcf()
fig.set_size_inches(150,100)


# In[59]:


dtg=DecisionTreeClassifier()
dtg.fit(x_train,y_train)
print("Gini Decision Tree Bias Score : ",dtg.score(x_train,y_train))
print("Gini Decision Tree Variance Score : ",dtg.score(x_test,y_test))


# In[60]:


#tree.plot_tree(dtg.fit(x_train,y_train),fontsize=6)


# In[61]:


dte=DecisionTreeClassifier(criterion="entropy")
dte.fit(x_train,y_train)
print("Entropy Decision Tree Bias Score : ",dte.score(x_train,y_train))
print("Entropy Decision Tree Variance Score : ",dte.score(x_test,y_test))


# In[62]:


#tree.plot_tree(dte.fit(x_train,y_train),fontsize=6)


# In[63]:


dtg1=DecisionTreeClassifier(criterion="gini",max_depth=10)
dtg1.fit(x_train,y_train)
print("Gini Decision Tree Bias Score : ",dtg1.score(x_train,y_train))
print("Entropy Decision Tree Variance Score : ",dtg1.score(x_test,y_test))


# In[64]:


#tree.plot_tree(dtg1.fit(x_train,y_train),fontsize=10)


# In[65]:


dte1=DecisionTreeClassifier(criterion="entropy",max_depth=10)
dte1.fit(x_train,y_train)
print("Entropy Decision Tree Bias Score : ",dte1.score(x_train,y_train))
print("Entropy Decision Tree Variance Score : ",dte1.score(x_test,y_test))


# In[66]:


#tree.plot_tree(dte1.fit(x_train,y_train),fontsize=10)


# # Implementing Random Forest Classifier

# In[67]:


from sklearn.ensemble import RandomForestClassifier


# In[68]:


rf=RandomForestClassifier(n_estimators=50,max_depth=10)
rf.fit(x_train,y_train)
print("Random Forest Bias Score : ",rf.score(x_train,y_train))
print("Random Forest Variance Score : ",rf.score(x_test,y_test))


# # Implementing K-Nearest Neighbours

# In[69]:


from sklearn.neighbors import KNeighborsClassifier


# In[70]:


knn=KNeighborsClassifier()
knn.fit(x_train,y_train)


# In[71]:


print("KNN Bias Score : ",knn.score(x_train,y_train))
print("KNN Variance Score : ",knn.score(x_test,y_test))


# In[72]:


y_predict=knn.predict(x_test)


# In[73]:


tbl=pd.DataFrame(list(zip(y_test,y_predict)),columns=['y_Observed','y_predicted'])
print(tbl)


# # Implementing Support Vector Machines

# In[74]:


from sklearn.svm import LinearSVC,SVC


# In[75]:


svc=LinearSVC()
svc.fit(x_train,y_train)
print("Linear SVC Bias Score : ",svc.score(x_train,y_train))
print("Linear SVC Variance Score : ",svc.score(x_test,y_test))

