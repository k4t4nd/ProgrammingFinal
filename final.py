#Import Dataset.
import pandas as pd
s = pd.read_csv("social_media_usage.csv")

print(s.shape)

#Create clean_sm to create binary values.
import numpy as np
def clean_sm(x):
    x = np.where(x==1,1,0)
    return(x)

#Test the function on toy dataset.
toy = {"Month":[1, 2, 3], "Num":[1, 1, 3]}
df_toy = pd.DataFrame(toy)
clean_toy = clean_sm(df_toy)
print(clean_toy)

#Add relevant columns from s to new dataframe.
ss = s.filter(["web1h","income","educ2","par","marital","gender","age"],axis=1)

#Rename columns.
ss.rename(columns={"web1h":"sm_li","par":"parent","educ2":"education","marital":"married","gender":"female"},inplace=True)

#Filter out missing values.
ss = ss[(ss["income"]<=9) & (ss["education"]<=8) & (ss["age"]<=98)]

#Change several variable types to binary. 
ss["sm_li"] = clean_sm(ss["sm_li"])
ss["female"] = np.where(ss["female"]==2,1,0)
ss["parent"]= clean_sm(ss["parent"])
ss["married"]= clean_sm(ss["married"])

#Final check for missing values.
ss.isnull().sum()
ss.head()

#Plot of Age versus being LinkedIn user.
import altair as alt
alt.Chart(ss).mark_bar().encode(
    x=alt.X('age', bin=True),
    y='count()',
    color = alt.Color("sm_li"))

#Scatterplot of probability of being LinkedIn user versus Income and Education.
alt.Chart(ss.groupby(["income", "education"], as_index=False)["sm_li"].mean()).\
mark_circle().\
encode(x="income",
      y="sm_li",
      color="education:N")

#Import packages for Machine Learning.
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

#Establish Target and Predictor Variables.
y = ss["sm_li"]
X = ss[["income","education","parent","married","female","age"]]

#Separate Train and Test Data.
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    stratify=y,      
                                                    test_size=0.2,   
                                                    random_state=123) 

#Initiate Regression Model.
lr = LogisticRegression(class_weight = "balanced")
lr.fit(X_train,y_train)

#Confusion Matrix
y_pred = lr.predict(X_test)
confusion_matrix(y_test, y_pred)

#Dataframe of Confusion Matrix
pd.DataFrame(confusion_matrix(y_test, y_pred),
            columns=["Predicted negative", "Predicted positive"],
            index=["Actual negative","Actual positive"]).style.background_gradient(cmap="PiYG")

#Manual Calculations of Precision, Recall, and F1
## recall: TP/(TP+FN)
recall = 68/(68+16)

## precision: TP/(TP+FP)
precision = 68/(68+57)

f1 = 2 * ((precision*recall)/(precision+recall))

print(f"The recall is {round(recall,2)}, the precision is {round(precision,2)}, and the F1 score is {round(f1,2)}.")

#Classification Report of metrics.
print(classification_report(y_test, y_pred))

person1 = [8,7,0,1,1,42]
person2 = [8,7,0,1,1,82]

# Predicting whether Person 1 is a LinkedIn user.
predicted_class = lr.predict([person1])

# Generating the probability associated with the predicted class.
probs = lr.predict_proba([person1])

print(f"Predicted class: {predicted_class[0]}") # 0=Not a LinkedIn user, 1=LinkedIn User
print(f"Probability that this person is a Linked In User: {round((probs[0][1]),2)}")

# Predicting whether Person 2 is a LinkedIn user.
predicted_class = lr.predict([person2])

# Generating the probability associated with the predicted class.
probs = lr.predict_proba([person2])

print(f"Predicted class: {predicted_class[0]}") # 0=Not a LinkedIn user, 1=LinkedIn User
print(f"Probability that this person is a Linked In User: {round((probs[0][1]),2)}")

#Moving onto Streamlit
import streamlit as st

st.title("Are you a LinkedIn user? Let me guess...")
st.header("Please answer the following questions.")

name = st.text_input("What is your name?")


income = st.selectbox("What is your houshold income?",["Less than $10,000","10 to under $20,000",
                                                   "20 to under $30,000","30 to under $40,000",
                                                   "40 to under $50,000","50 to under $75,000",
                                                   "75 to under $100,000","100 to under $100,000",
                                                   "$150,000 or more"], placeholder="Choose an option",index=None)

if income == "Less than $10,000":
    income = 1
elif income == "10 to under $20,000":
    income = 2
elif income == "20 to under $30,000":
    income = 3
elif income == "30 to under $40,000":
    income = 4
elif income == "40 to under $50,000":
   income = 5
elif income == "50 to under $75,000":
    income = 6
elif income== "75 to under $100,000":
    income = 7
elif income == "100 to under $100,000":
    income = 8
else:
    income = 9

education = st.selectbox("What is your highest education level?",["Less than High School (Grades 1-8 or no formal schooling)",
                                                                  "High school incomplete (Grades 9-11 or Grade 12 with NO diploma)",
                                                                  "High school graduate (Grade 12 with diploma or GED certificate)",
                                                                  "Some college, no degree (includes some community college)",
                                                                  "Two-year associate degree from a college or university",
                                                                  "Four-year college or university degree/Bachelor's degree (e.g., BS, BA, AB)",
                                                                  "Some postgraduate or professional schooling, no postgraduate degree (e.g. some graduate school),",
                                                                  "Postgraduate or professional degree, including master's, doctorate, medicate or law degree (e.g., MA, MS, PhD, MD, JD)"],
                                                                  placeholder="Choose an option",index=None)

if education == "Less than High School (Grades 1-8 or no formal schooling)":
    education = 1
elif education == "High school incomplete (Grades 9-11 or Grade 12 with NO diploma)":
    education = 2
elif education == "High school graduate (Grade 12 with diploma or GED certificate)":
    education = 3
elif education == "Some college, no degree (includes some community college)":
    education = 4
elif education == "Two-year associate degree from a college or university":
    education = 5
elif education ==  "Four-year college or university degree/Bachelor's degree (e.g., BS, BA, AB)":
    education = 6
elif education== "Some postgraduate or professional schooling, no postgraduate degree (e.g. some graduate school),":
   education = 7
else:
    education = 8


parent = st.radio("Are you a parent?",["Yes","No"],index=None)

if parent == "Yes":
    parent = 1
else: 
    parent = 0

married = st.radio("Are you married?",["Yes","No"],index=None)

if married == "Yes":
    married = 1
else: 
    married = 0

female = st.radio("Are you a female?",["Yes","No"],index=None)

if female == "Yes":
    female = 1
else: 
    female = 0

age = st.number_input("How old are you?",min_value=0, max_value=98)

user_input = [income, education, parent, married, female, age]

#st.button("Predict whether I use LinkedIn")

result = lr.predict([user_input])
result_prob = lr.predict_proba([user_input])

if 'clicked' not in st.session_state:
    st.session_state.clicked = False

def click_button():
    st.session_state.clicked = True

st.button('Predict :sparkles:', on_click=click_button)

if (st.session_state.clicked) & (result[0]==1):
    st.subheader(f"{name}, you are likely a LinkedIn user! The probability of you using LinkedIn is {(round((result_prob[0][1]),2))}.")
    col1, col2 = st.columns(2)
    with col1:
        st.image("gtown.jpg",width=300)
    with col2:
        st.text("\nThis app was created by a Programming II \nstudent as part of Georgetown's Master \nof Science in Business Analytics program. \nUsing a dataset with features for social \nmedia usage and other predictory \ndemographics, the app applies logistic \nregression to determine the \nprobability of an individual being a \nLinkedIn user. I hope you are happy with \nyour results!")
elif(st.session_state.clicked) & (result[0]==0):
    st.subheader(f"{name}, you aren't likely to be a LinkedIn user. The probability of you using LinkedIn is {(round((result_prob[0][1]),2))}.")
    col1, col2 = st.columns(2)
    with col1:
        st.image("gtown.jpg",width=300)
    with col2:
        st.text("\nThis app was created by a Programming II \nstudent as part of Georgetown's Master \nof Science in Business Analytics program. \nUsing a dataset with features for social \nmedia usage and other predictory \ndemographics, the app applies logistic \nregression to determine the \nprobability of an individual being a \nLinkedIn user. I hope you are happy with \nyour results!")