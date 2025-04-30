from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,precision_score,recall_score,roc_auc_score,accuracy_score
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
dataset=pd.read_csv("Titanic-Dataset.csv")
inf=dataset.info()
x=dataset[['Pclass','Age','SibSp','Parch','Fare']]
y=dataset['Survived']
x.fillna(x.mean(),inplace=True)
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
model=LogisticRegression()
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
cm=confusion_matrix(y_test,y_pred)
print(cm)
precision=precision_score(y_test,y_pred)
print(f"Precision score is {precision}")
recall=recall_score(y_test,y_pred)
print(f"Recall score is {recall}")
roc_auc=roc_auc_score(y_test,y_pred)
print(f"roc_auc is {roc_auc}")
accuracy=accuracy_score(y_test,y_pred)
print(f"Accuracy is {accuracy}")
new_input=np.array([[2,4,5,9,10]])
new_pred=model.predict(new_input)
print(f"Predicted new value is {new_pred}")
for i in dataset.select_dtypes(include="number"):
   sns.histplot(x=y_test,y=y_pred)
   plt.xlabel("Actual price")
   plt.ylabel("Predicted price")
   plt.show()


