from sklearn.datasets import fetch_openml
import pandas as pd

# ----- Fetch the 'adult' dataset from openml as a pandas DataFrame
adult = fetch_openml(name="adult", version=2, as_frame=True)
df = adult.frame                                       # get single DataFrame (data + target combined)
# print(df.shape)
# print(df.head())
# print(df.info())
# print(df['class'].value_counts())                      # show label distribution (<=50K, >50K)


# ----- Complete missing data with a default value
import numpy as np

df = df.replace('?', np.nan)
# print(df.isna().sum())  # see counts of missing per column

# fill categorical missing with mode
for col in ['workclass', 'occupation', 'native-country']:
    df[col] = df[col].fillna(df[col].mode()[0])      # replace NaN with most frequent category
# print(df.isna().sum())


# ----- split features and target
y = df["class"].map({"<=50K": 0, ">50K": 1})

features = [    # select only relevant features
    'age', 'workclass', 'education', 'marital-status', 'occupation',
    'relationship', 'race', 'sex', 'capital-gain', 'capital-loss',
    'hours-per-week', 'native-country'
]
x = df[features].copy()


# ----- Split the dataset into test/train sets
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)

# ----- Replace categories with numbers
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# lists of column names by type
num_cols = ['age', 'capital-gain', 'capital-loss', 'hours-per-week']
cat_cols = ['workclass','education','marital-status','occupation','relationship','race','sex','native-country']

num_transformer = StandardScaler() # scale numeric columns to mean=0 std=1
cat_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=False) # converts categorical values into 0/1 vectors

# ColumnTransformer: apply num_transformer to num_cols, cat_transformer to cat_cols
preprocessor = ColumnTransformer(
    transformers=[
        ('num', num_transformer, num_cols),
        ('cat', cat_transformer, cat_cols)
    ]
)
# preprocessor = a single array ready for any ML model


# ----------------------------------- Logistic Regression --------------------------------------------
# ----- Train the model
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

log_pipe = Pipeline([
    ('pre', preprocessor),
    ('clf', LogisticRegression(max_iter=1000, solver='lbfgs'))
])

log_pipe.fit(x_train, y_train)


# ----------------------------------- Random Forest --------------------------------------------
# ----- Create Decision Trees as a model
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

tree_pipe = Pipeline([('pre', preprocessor), ('clf', DecisionTreeClassifier(max_depth=6))])
rf_pipe = Pipeline([('pre', preprocessor), ('clf', RandomForestClassifier(n_estimators=100))])
rf_pipe.fit(x_train, y_train)


# ----- Visualize the importance of each feature
feature_names_num = num_cols
cat_all_features = rf_pipe.named_steps['pre'].named_transformers_['cat']
cat_feature_names = list(cat_all_features.get_feature_names_out(cat_cols))
feature_names = feature_names_num + cat_feature_names

importances = rf_pipe.named_steps['clf'].feature_importances_

# Pair feature names with their importance values
feat_importances = pd.DataFrame({
    "feature": feature_names,
    "importance": importances
})

print("The importance of each feature when using Decision Trees:")
print(feat_importances.sort_values("importance", ascending=False).head(15))
print()
# ----- Evaluate both models (Logistic Regression and Random Forest)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score

def evaluate_model(pipe, x_test, y_test, name):
    y_pred = pipe.predict(x_test)
    y_prob = pipe.predict_proba(x_test)[:, 1]
    acc = accuracy_score(y_test, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="binary")
    auc = roc_auc_score(y_test, y_prob)
    return {"Model": name, "Accuracy": acc, "Precision": prec, "Recall": rec, "F1": f1, "AUC": auc}

results = []
results.append(evaluate_model(log_pipe, x_test, y_test, "Logistic Regression"))
results.append(evaluate_model(rf_pipe, x_test, y_test, "Random Forest"))

# Print results as a clean comparison table
results_df = pd.DataFrame(results)
print("The overall differences between the 2 models:")
print(results_df)
