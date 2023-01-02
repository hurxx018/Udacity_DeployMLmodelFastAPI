# Model Card

## Model Details
Kwang Ho Hur created the model that is random forest.\
Grid search was used with cross-validation using hyperparameters below:\
n_estimators    : [5, 8]\
max_features    : ['auto', 'sqrt']\
max_depth       : [4, 5, 7]\
criterion       : ['gini', 'entropy']

## Intended Use
This model should be used to predict whether an annual income of a person exceeds $50K based on census date with the following attributes:\
age, workclass, fnlgt, education, education-num, marital-status, occupation, \
relationship, race, sex, capital-gain, capital-loss, hours-per-week, native-country

## Training and Test Data
The origin of data is Census Income Data Set that is obtained from the UCI Machine Learning Repository (https://archive.ics.uci.edu/ml/datasets/census+income). The data was messy including white spaces. This was cleaned before the development of the model.

The original data has 32561 rows. The data was split to the train and test sets with a ratio of 80-20. No stratification was done. One-hot encoder was used to on the categorical features. A label binarizer was used on the label.

## Metrics
The model was evaluated on the test set using precision, recall, and f1 beta score. Their values are 0.77, 0.61, and 0.68, respectively.

## Ethical Considerations
Occupation categories are limited to be one of the following:\
'Adm-clerical', 'Exec-managerial', 'Handlers-cleaners',\
'Prof-specialty', 'Other-service', 'Sales', 'Transport-moving',\
'Farming-fishing', 'Machine-op-inspct', 'Tech-support',\
'Craft-repair', 'Protective-serv', 'Armed-Forces',\
'Priv-house-serv'

Some people might be out of this category.

## Caveats and Recommendations
The data was created from 1994 Census database, so capital-gain and captial-loss must be adjused based on the inflation.