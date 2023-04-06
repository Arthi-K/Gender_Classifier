!pip install gender_guesser
# create a dataset
import gender_guesser.detector as gender
import pandas as pd
import numpy as np
# ML Packages
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import DictVectorizer
odf = pd.read_csv('author_names_unclassified.csv')
def myFunction(name):
    # process name here
    first_name = name.split(' ')[0]
    if len(first_name) > 1: 
        return name
    return 'TODO'
  

# remove single first names
odf['new_author_name'] = odf['Author Name'].apply(myFunction)
df = odf[odf['new_author_name'] != 'TODO']
# Remove duplicates in 'Name' column and create new dataframe
new_df = df.drop_duplicates(subset=['Author Name']).copy()

# Create a new column that contains the first name of each author
new_df['First Name'] = new_df['Author Name'].str.split().str.get(0)

d = gender.Detector()
conv = {'male': 'M', 'mostly_male': 'M', 'female': 'F', 'mostly_female': 'F', 'andy': 'M', 'unknown': 'unknown'}
def classify_gender(name):
    if isinstance(name, str):
        first_name = name.split()[0]
        return conv[d.get_gender(first_name)]
    else:
        return 'unknown'
# Apply the function to the author names and create a new column with the gender classification
new_df['sex'] = new_df['First Name'].apply(classify_gender)

new_df['name'] = new_df['First Name']

# final_df = new_df['name', 'sex']
final_df = new_df.loc[:, ['name', 'sex']].copy()
df = final_df # for easier mapping

# Data Cleaning
# Checking for column name consistency
df.columns

# Data Types
df.dtypes
# Checking for Missing Values
df.isnull().isnull().sum()
df['sex'].unique()
# remove unknowns.
df = df[df['sex'] != 'unknown']
df['sex'].unique()

# Number of Female Names
df[df.sex == 'F'].size
# Number of Male Names
df[df.sex == 'M'].size

df_names = df
# Replacing All F and M with 0 and 1 respectively
df_names.sex.replace({'F':0,'M':1},inplace=True)
df_names.sex.unique()
df_names.dtypes
Xfeatures =df_names['name']
# Feature Extraction 
cv = CountVectorizer()
X = cv.fit_transform(Xfeatures)
cv.get_feature_names()
from sklearn.model_selection import train_test_split

# Features 
X
# Labels
y = df_names.sex
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
# Naive Bayes Classifier
from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB()
clf.fit(X_train,y_train)
clf.score(X_test,y_test)

# Accuracy of our Model
print("Accuracy of Model",clf.score(X_test,y_test)*100,"%")
# Sample1 Prediction
sample_name = ["Mary"]
vect = cv.transform(sample_name).toarray()
# Female is 0, Male is 1
clf.predict(vect)
# Sample2 Prediction
sample_name1 = ["Mark"]
vect1 = cv.transform(sample_name1).toarray()
clf.predict(vect1)
# Sample3 Prediction of Russian Names
sample_name2 = ["Natasha"]
vect2 = cv.transform(sample_name2).toarray()
clf.predict(vect2)
# Sample3 Prediction of Random Names
sample_name3 = ["Nefertiti","Nasha","Ama","Ayo","Xhavier","Ovetta","Tathiana","Xia","Joseph","Xianliang"]
vect3 = cv.transform(sample_name3).toarray()
clf.predict(vect3)
# A function to do it
def genderpredictor(a):
    test_name = [a]
    vector = cv.transform(test_name).toarray()
    if clf.predict(vector) == 0:
        return "F"
    else:
        return ("M")
genderpredictor("Martha")
# genderpredictor("karthik")
namelist = ["Arti", "Martha", "Yaa","Yaw","Femi","Masha"]
for i in namelist:
    print(genderpredictor(i))
new_df[new_df['sex'] == 'unknown']
for index, row in new_df.iterrows():
    if row['sex'] == 'unknown':
        pred = genderpredictor(row["First Name"])
        fname = row["First Name"]
        if pred != None:
            print(f"Predicting {fname} : {pred}" )
            new_df.at[index, 'sex'] = pred
new_df['sex'].unique()
new_df.to_csv('predicted_data_v1.csv')
