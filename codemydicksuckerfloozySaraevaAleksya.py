#!/usr/bin/env python
# coding: utf-8

# In[ ]:


X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.3, random_state=1, stratify=y )
from sklearn.preprocessing import StandardScaler 
sc = StandardScaler() 
sc.fit(X_train) 
X_train_std = sc.transform(X_train) 
X_test_std = sc.transform(X_test)
X_test_std = sc.transform(X_test) 
y_pred = ppn.predict(X_test_std) 
print('Misclassified examples: %d' % (y_test != y_pred).sum())
from sklearn.metrics import accuracy_score 
print('Accuracy: %.3f' % accuracy_score(y_test, y_pred))
from sklearn.linear_model import LogisticRegression 
lr = LogisticRegression(C=100.0, solver='lbfgs', multi_class='ovr') 
lr.fit(X_train_std, y_train)

