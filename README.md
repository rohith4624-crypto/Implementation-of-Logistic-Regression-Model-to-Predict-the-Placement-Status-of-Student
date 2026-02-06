<img width="786" height="562" alt="image" src="https://github.com/user-attachments/assets/d1d59e7f-e7c3-44fc-8475-998dd1ce778a" /># Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required packages and print the present data.
2. Print the placement data and salary data
3. Find the null and duplicate values.
4. Using logistic regression find the predicted values of accuracy , confusion matrices.
5. Display the results. 

## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: ROHITH R
RegisterNumber:  25002211
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("Placement_Data.csv")
print(data.head())

data1 = data.copy()

data1.drop(['sl_no', 'salary'], axis=1, inplace=True)

print("\nMissing values:\n", data1.isnull().sum())
print("\nDuplicate values:", data1.duplicated().sum())

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
data1['gender'] = le.fit_transform(data1['gender'])
data1['ssc_b'] = le.fit_transform(data1['ssc_b'])
data1['hsc_b'] = le.fit_transform(data1['hsc_b'])
data1['hsc_s'] = le.fit_transform(data1['hsc_s'])
data1['degree_t'] = le.fit_transform(data1['degree_t'])
data1['workex'] = le.fit_transform(data1['workex'])
data1['specialisation'] = le.fit_transform(data1['specialisation'])
data1['status'] = le.fit_transform(data1['status'])

x = data1.iloc[:, :-1]
y = data1['status']

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=0
)

from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(solver='liblinear')
lr.fit(x_train, y_train)

y_pred = lr.predict(x_test)

from sklearn.metrics import accuracy_score
print("\nAccuracy:", accuracy_score(y_test, y_pred))

from sklearn.metrics import confusion_matrix
confusion = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:\n", confusion)

from sklearn.metrics import classification_report
print("\nClassification Report:\n", classification_report(y_test, y_pred))

from sklearn import metrics

cm_display = metrics.ConfusionMatrixDisplay(
    confusion_matrix=confusion,
    display_labels=['Not Placed', 'Placed']
)

cm_display.plot()
plt.show()
*/
```

## Output:
<img width="314" height="73" alt="Screenshot 2026-02-06 111050" src="https://github.com/user-attachments/assets/80eff5d1-c9b0-4f09-8cff-548ef4ff1b4e" />

<img width="271" height="115" alt="Screenshot 2026-02-06 111141" src="https://github.com/user-attachments/assets/28d82bfa-bdbd-4abd-8210-1c6a7f488c4f" />
<img width="534" height="234" alt="Screenshot 2026-02-06 111218" src="https://github.com/user-attachments/assets/8e1f9577-5736-4d35-a91c-7540a8c0b6ab" />


<img width="786" height="562" alt="Screenshot 2026-02-06 111301" src="https://github.com/user-attachments/assets/5bca7df2-f539-49bb-aa2d-57686ac1df45" />


## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
