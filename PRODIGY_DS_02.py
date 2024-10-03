
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
titanic_data = pd.read_csv(r"D:\Downloads\test.csv")




print(titanic_data.shape)




titanic_data.head()




print(titanic_data.describe())




print(titanic_data.info())




# Check for missing values
print(titanic_data.isnull().sum())




# Fill missing values in 'Age' with the median value
titanic_data['Age'].fillna(titanic_data['Age'].median(), inplace=True)




# Drop the 'Cabin' column if it's mostly missing
titanic_data.drop(columns=['Cabin'], inplace=True)





# Drop rows with missing 'Embarked' values (if any)
titanic_data.dropna(subset=['Embarked'], inplace=True)




# Remove duplicate rows
titanic_data.drop_duplicates(inplace=True)




# Value counts of 'Sex'
print(titanic_data['Sex'].value_counts())





# Value counts of 'Pclass'
print(titanic_data['Pclass'].value_counts())




# Value counts of 'Embarked'
print(titanic_data['Embarked'].value_counts())




# Distribution of 'Age'
sns.histplot(titanic_data['Age'],color='blue')
plt.title('Age Distribution')
plt.show()




# Distribution of 'Fare'
sns.histplot(titanic_data['Fare'], color='yellow')
plt.title('Fare Distribution')
plt.show()




# Count plot for 'Pclass'
sns.countplot(x='Pclass', data=titanic_data,palette="Set3")
plt.title('Passenger Class Distribution')
plt.show()




# Count plot for 'Sex'
sns.countplot(x='Sex', data=titanic_data, palette="Set1")
plt.title('Gender Distribution')
plt.show()




# Boxplot of 'Fare' by 'Pclass'
sns.boxplot(x='Pclass', y='Fare', data=titanic_data)
plt.title('Fare by Passenger Class')
plt.show()




# Boxplot of 'Age' by 'Pclass'
sns.boxplot(x='Pclass', y='Age', data=titanic_data, palette="Set2")
plt.title('Age by Passenger Class')
plt.show()




# Boxplot of 'Fare' by 'Sex'
sns.boxplot(x='Sex', y='Fare', data=titanic_data, palette="Set3")
plt.title('Fare by Gender')
plt.show()




# Correlation matrix
corr_matrix = titanic_data.corr(numeric_only=True)
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()




# Boxplot to check for outliers in 'Fare'
sns.boxplot(x=titanic_data['Fare'],palette="Set1")
plt.title('Fare Outliers')
plt.show()





# Boxplot to check for outliers in 'Age'
sns.boxplot(x=titanic_data['Age'],palette="Set3")
plt.title('Age Outliers')
plt.show()




# Create a new column for family size
titanic_data['FamilySize'] = titanic_data['SibSp'] + titanic_data['Parch'] + 1

# Check the distribution of the new feature
sns.countplot(x='FamilySize', data=titanic_data, palette="Set2")
plt.title('Family Size Distribution')
plt.show()




# Pair plot of selected numerical features
sns.pairplot(titanic_data[['Age', 'Fare', 'Pclass']], hue='Pclass', palette="Set1")
plt.show()





