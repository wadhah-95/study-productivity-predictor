import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib

df=pd.read_csv("../data/study-sessions.csv")
#Identifying features vs target
X=df.drop(["productive"], axis=1)
Y=df["productive"]
#One Hot-Encoding
X_encoded=pd.get_dummies(X, columns=["day_type"], drop_first=True)
#split train/test data
X_train, X_test, Y_train, Y_test=train_test_split(X_encoded, Y, test_size=0.2, random_state=42, stratify=Y)
#Train first model
model=LogisticRegression()
model.fit(X_train, Y_train)
#make predictions
Y_predict=model.predict(X_test)
#evaluate accuracy
accuracy=accuracy_score(Y_test, Y_predict)
print("The accuracy is: \t", accuracy)
print("Classification Report: \n", classification_report(Y_test, Y_predict))
#Saving model into models folder using joblib library
joblib.dump(model, "../models/study_productivity_model.joblib")
print("Model saved successfully")

