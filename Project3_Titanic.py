import pandas as pd

# ----- Load the data and visualise it
df = pd.read_csv("train.csv")
# print(df.head())
# print(df.info())
# print(df.describe())


# ----- Get rid of useless columns
df = df.drop(columns=["Name", "Ticket", "Cabin"])


# ----- Complete missing data with a default variable
df["Age"] = df["Age"].fillna(df["Age"].median())
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])


# ----- Replace any categories with numbers (ML need numbers, not strings)
df["Sex"] = df["Sex"].map({"male": 0, "female": 1})
df = pd.get_dummies(df, columns=["Embarked"], drop_first=True) # make each of the categories a boolean column
                                                               # (no need for 3 columns, since with 2 we know all)

# ----- Split features and target
X = df.drop(columns=["Survived"]) # X = all the table without "Survived" column
y = df["Survived"]                # y = the predictions that we know from the dataset for "Survived"


# ----- Split the dataset into test/train sets
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# ----- Train the model
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)


# ----- Evaluate the accuracy
from sklearn.metrics import accuracy_score

y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))



