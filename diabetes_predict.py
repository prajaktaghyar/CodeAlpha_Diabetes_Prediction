#!/usr/bin/env python
# coding: utf-8

# In[23]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


# In[24]:


df = pd.read_csv('diabetes.csv')


# In[25]:


features = ['Glucose', 'BloodPressure', 'BMI', 'Age']
X = df[features]
y = df['Outcome']


# In[26]:


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# In[27]:


X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=0)


# In[28]:


model = LogisticRegression()
model.fit(X_train, y_train)


# In[29]:


y_pred = model.predict(X_test)
print("✅ Accuracy:", accuracy_score(y_test, y_pred))
print("🔍 Classification Report:\n", classification_report(y_test, y_pred))


# In[30]:


print("\n📥 Enter patient details:")
glucose = float(input("Glucose level: "))
bp = float(input("Blood Pressure: "))
bmi = float(input("BMI: "))
age = float(input("Age: "))


# In[31]:


# Add feature names when transforming
input_dict = {
    'Glucose': [glucose],
    'BloodPressure': [bp],
    'BMI': [bmi],
    'Age': [age]
}

patient_df = pd.DataFrame(input_dict)
patient_scaled = scaler.transform(patient_df)


# In[32]:


result = model.predict(patient_scaled)


# In[33]:


print("\n📊 Prediction:")
if result[0] == 1:
    print("🔴 The patient is likely to have diabetes.")
else:
    print("🟢 The patient is unlikely to have diabetes.")


# In[ ]:




