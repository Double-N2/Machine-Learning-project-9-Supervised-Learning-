import pandas as pd
import psutil as ps
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import ElasticNet
from sklearn.compose import ColumnTransformer
from sklearn.metrics import r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline


# Reading and Cleaning the data

sales = pd.read_csv("sales.csv")
customers = pd.read_csv("customers.csv")
products = pd.read_csv("products.csv")

# print(sales.columns)
# print(customers.columns)
# print(products.columns)

# Merging the datasets
data = sales.merge(customers,on="Customer_ID")
data = data.merge(products, on='Product_ID')
print(data.columns)

data.drop(['Customer_ID','Order_ID','Product_ID'], axis=1,inplace=True)
data.drop_duplicates(inplace=True)
data.reset_index(drop=True, inplace=True)
pd.set_option('display.max_columns',None)
data.info()
print(data.head())

Number_only = data.select_dtypes(include = 'number').columns # To select only numbers and float in one column

Characters_only = data.select_dtypes(include = ['object','string']).columns # To select only characters


# Converting the data into Computer Understandable format using a Column transformer

preprocessing = ColumnTransformer(
    transformers=[(
        'Numbers_only',
        Pipeline([
            ('simple', SimpleImputer(strategy='median')),
            ('encoder',StandardScaler())
        ]),
     Number_only
    ),
        (
            'Characters_only',
            Pipeline([
                ('simple', SimpleImputer(strategy='most_frequent')),
                ('encoder',OneHotEncoder(handle_unknown='ignore'))
            ]),
            Characters_only
        )
    ])


y = data['Total_Reviews']
x = data.drop('Total_Reviews', axis=1)

# Splitting the model using train test split

x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=42)

# Using a Pipeline

Linear_pipeline = Pipeline([
    ('preprocessing', preprocessing),
    ('Linear', LinearRegression())
])
Gradient_Boosting_pipeline = Pipeline([
    ('preprocessing', preprocessing),
    ('Gradient', GradientBoostingRegressor())
])
RandomForest_regression_pipeline = Pipeline([
    ('preprocessing', preprocessing),
    ('Random', RandomForestRegressor())
])
Elasticnet_pipeline = Pipeline([
    ('preprocessing', preprocessing),
    ('Elastic_Net', ElasticNet())
])
DecisionTree_regression_pipeline = Pipeline([
    ('preprocessing', preprocessing),
    ('DecisionTree', DecisionTreeRegressor())
])

# Creating hyperparameters
linear_regression_parameters = {
    "Linear__fit_intercept": [True, False]
}

Gradient_Boosting_parameters = {
    "Gradient__n_estimators": [50, 100],        # fewer trees
    "Gradient__learning_rate": [0.1],           # single value
    "Gradient__max_depth": [3, 5]               # shallow trees
}

RandomForest_regression_parameters = {
    "Random__n_estimators": [50, 100],        # fewer trees
    "Random__max_depth": [None, 10],          # limit depth
    "Random__min_samples_split": [2, 5]
}

ElasticNet_parameters = {
    "Elastic_Net__alpha": [0.1, 1.0],              # small set
    "Elastic_Net__l1_ratio": [0.5]                 # fixed balance
}

DecisionTree_regressor_parameters = {
    "DecisionTree__criterion": ["squared_error"],   # single criterion
    "DecisionTree__max_depth": [None, 5, 10],       # shallow depth
    "DecisionTree__min_samples_split": [2, 5]
}

# Using Grid Search Cross Validation
linear_grid = GridSearchCV(
    model = Linear_pipeline,
    param_grid = linear_regression_parameters,
    scoring = "r2",
    cv = 5
)

Gradient_Boosting_grid = GridSearchCV(
    model = Gradient_Boosting_pipeline,
    param_grid = Gradient_Boosting_parameters,
    scoring = "r2",
    cv = 5
)
RandomForest_grid = GridSearchCV(
    model = RandomForest_regression_pipeline,
    param_grid = RandomForest_regression_parameters,
    scoring = "r2",
    cv = 5
)
Elasticnet_grid = GridSearchCV(
    model = Elasticnet_pipeline,
    param_grid = ElasticNet_parameters,
    scoring = "r2",
    cv = 5
)
DecisionTree_grid = GridSearchCV(
    model = DecisionTree_regression_pipeline,
    param_grid = DecisionTree_regressor_parameters,
    scoring = "r2",
    cv = 5
)

linear_grid = linear_grid.fit(x_train, y_train)
Gradient_Boosting_grid = Gradient_Boosting_grid.fit(x_train, y_train)
RandomForest_grid = RandomForest_grid.fit(x_train, y_train)
Elasticnet_grid = Elasticnet_grid.fit(x_train, y_train)
DecisionTree_grid = DecisionTree_grid.fit(x_train, y_train)

# Predicting using the Total Reviews

linear = linear_grid.predict(x_test)
Gradient_Boosting = Gradient_Boosting_grid.predict(x_test)
RandomForest = RandomForest_grid.predict(x_test)
Elastic_net = Elasticnet_grid.predict(x_test)
DecisionTree = DecisionTree_grid.predict(x_test)

# Calculating the accuracy of the models

linear_accuracy = r2_score(y_test, linear)
Gradient_accuracy = r2_score(y_test, Gradient_Boosting)
RandomForest_accuracy = r2_score(y_test, RandomForest)
Elastic_net_accuracy = r2_score(y_test, Elastic_net)
DecisionTree_accuracy = r2_score(y_test, DecisionTree)


print("\n")
print("Best Parameters for Linear Regression: :",linear_grid.best_params_)
print("Best score for Linear Regression is: ",linear_grid.best_score_)
print("Linear Regression: ",linear)
print("Linear Regression Accuracy: ",linear_accuracy)
print("\n")
print("Best Parameter for Gradient Boosting: ",Gradient_Boosting_grid.best_params_)
print("Best Score for Gradient Boosting is: ",Gradient_Boosting_grid.best_score_)
print("Gradient Boosting: ",Gradient_Boosting)
print("Gradient Boosting Accuracy: ",Gradient_accuracy)
print("\n")
print("Best Parameters for Random Forest: ",RandomForest_grid.best_params_)
print("Best score for Random Forest is: ",RandomForest_grid.best_score_)
print("Random Forest: ",RandomForest)
print("Random Forest Accuracy: ",RandomForest_accuracy)
print("\n")
print("Best Parameters for Elastic Net: ",Elastic_net.best_params_)
print("Best score for Elastic Net is: ",Elastic_net.best_score_)
print("Elastic net: ",Elastic_net)
print("Elastic Net Accuracy: ",Elastic_net_accuracy)
print("\n")
print("Best Parameters for DecisionTree: ",DecisionTree.best_params_)
print("Best score for DecisionTree is: ",DecisionTree.best_score_)
print("DecisionTree: ",DecisionTree)
print("DecisionTree Accuracy: ",DecisionTree_accuracy)

# Determining the Computer Performance usage like Disk, RAM and CPU

print("\n")
print("Disk usage",ps.disk_usage('%').percent)
print("RAM usage: ",ps.virtual_memory().percent)
print("CPU usage: ",ps.cpu_percent())
