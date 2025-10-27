import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, mean_absolute_error, mean_squared_error, classification_report)
import matplotlib.pyplot as plt
import seaborn as sns

file="D:\\Notes\\Career\\Data Science\\Skills\\Python\\Projects\\Elevate Labs (Internship)\\Project Phase\\Healthcare Appointment No-Show Prediction\\Healthcare Appointment.csv"
df=pd.read_csv(file)
#print(df.head(5))  #Analyzing the dataset
#print(df.isnull().sum().sum()) #-0, calculating the null values in the dataset
#print(df.info()) #Insights of the non-null columns and their data types
#print(df.describe()) #Insights of the mathematical details of the numeral columns
#print(df.shape)   -(106987, 15)  #Total rows and columns
#df.drop_duplicates(inplace=True)  #Eliminating duplicates from the dataset
#print(df.shape) #Verifying again if there were any duplicates in the dataset originally
df.columns = df.columns.str.strip().str.title()
#print(df.columns) #Refining the column names to be in the Title case with no trailing or leading spaces present

#Label Encoding on the columns to turn them from boolean to int values for ML model.
le=LabelEncoder()
#print(df.columns)
df['Scholarshipped']= le.fit_transform(df['Scholarship'])
df['Hipertensed']= le.fit_transform(df['Hipertension'])
df['Diabetic']= le.fit_transform(df['Diabetes'])
df['Alcoholic']= le.fit_transform(df['Alcoholism'])
df['Handicapped']= le.fit_transform(df['Handicap'])
df['SMS']= le.fit_transform(df['Sms_Received'])
df['Showed']= le.fit_transform(df['Showed_Up']) #When Showed_Up is True, then Showed=1

#print(df.columns)
#print(df.head(3))
#print(df.info())

#print(df['Gender'].unique())  #Fetching all the unique data values present in the 'Gender' column.
#print(df['Neighbourhood'].unique())  #Fetching all the unique data values present in the 'Neighbourhood' column.
#print(df['Neighbourhood'].value_counts())  #Counting the occurrences of the unique values present in the 'Neighbourhood' column

#I have a dataframe column having 81 unique values and multiple copies of the same values for 10,000 cells.
# I need to display top 20 values as their names and rest as "others". How can I achieve it

#Naming the neighbourhood names as 'others' for the ones which are outside the top 30 by their patient frequency

#1) Get top 30 most frequent values
#print(df.columns.tolist())
top_30 = df['Neighbourhood'].value_counts().nlargest(30).index

# Step 2: Replace all other values with 'Others'
df['Location'] = df['Neighbourhood'].where(df['Neighbourhood'].isin(top_30), 'Others')

#One Hot Encoding for the 'Neighbourhood' and 'Gender' columns using the get dummies of Pandas
df = pd.get_dummies(df, columns=['Neighbourhood', 'Gender'])

#Fetching the overall 'No-Show' rate
percentages = df['Showed'].value_counts(normalize=True) * 100
#print(percentages) #No show rate- 20.2%

#pd.set_option('display.max_columns', None)
#print(df.head(2))

#No show by SMS received- Fetching the rate of no shows when grouped by the SMS being received or not
show_rate_by_sms = df.groupby('SMS')['Showed'].mean() * 100
#print(show_rate_by_sms)

#SMS reminder- False, show percentage- 83.271180, SMS reminder- True, show percentage- 72.334827

#Show percentage by Age
show_rate_by_age = df.groupby('Age')['Showed'].mean() * 100
#print(show_rate_by_age)



#Building the ML model- Decision Tree Classifier in this case on the basis of above data
# Features to take into account for the ML model to train
X=df[['Age','Scholarshipped','Hipertensed','Diabetic','Alcoholic','Handicapped','SMS','Date_Diff']]

#Prediction by the model for the column
y= df[['Showed']]

#Splitting overall data into train and test groups for the model in the ratio- 80% to 20%
X_train, X_test, y_train, y_test= train_test_split(X,y, test_size=0.2, random_state=42)
#print("Training Data")
#print(X_train)

#print("Testing Data")
#print(X_test)

#Creating an object of the DecisionTreeClassifier
model= DecisionTreeClassifier()

#Training the model based on the dataframe data
model.fit(X,y)

#Now comes the prediction part by the model, taking the inputs from the user and letting the model predict and display output based on the training data
Patient_Age= input("Enter the age of the patient")
Patient_Scholarshipped= int(input("Enter 1 if patient opted for scholarship, enter 0 if not opted"))
Patient_Hipertensed= int(input("Enter 1 if patient was suffering from hypertension, enter 0 if not"))
Patient_Diabetic= int(input("Enter 1 if patient was suffering from diabetes, enter 0 if not"))
Patient_Alcoholic= int(input("Enter 1 if patient was Alcoholic, enter 0 if not"))
Patient_Handicapped= int(input("Enter 1 if patient was handicapped, enter 0 if not"))
Patient_SMS= int(input("Enter 1 if patient received the reminder SMS, enter 0 if not"))
Patient_Date_Diff= int(input("Enter the days between the scheduled date and appointment date"))

prediction= model.predict([[Patient_Age, Patient_Scholarshipped, Patient_Hipertensed, Patient_Diabetic, Patient_Alcoholic, Patient_Handicapped, Patient_SMS, Patient_Date_Diff]])[0]
if prediction==0:
     print("The patient is unlikely to show up on the appointment day")
else:
	 print("The patient is likely to show up on the appointment day")


#Model Evaluation-

#True answers data (what actually happened)
y_true= df['Showed']

#Predicted answers by the model
y_pred=model.predict(df[['Age','Scholarshipped','Hipertensed','Diabetic','Alcoholic','Handicapped','SMS','Date_Diff']])

#evaluation
print("Accuracy: ", accuracy_score(y_true, y_pred))
print("Precision: ", precision_score(y_true, y_pred))
print("Recall: ", recall_score(y_true, y_pred))
print("F1 Score: ", f1_score(y_true, y_pred))
print("Detailed Report : ", classification_report(y_true, y_pred))
print("MAE: ", mean_absolute_error(y_true, y_pred))
print("MSE: ", mean_squared_error(y_true, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
print("Confusion Matrix (numeric):\n", cm)
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Showed", "No-Show"],
            yticklabels=["Showed", "No-Show"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - Decision Tree Model")
plt.show()