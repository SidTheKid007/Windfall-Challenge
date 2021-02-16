<h1 align="center" style="font-weight:bold;font-size:32px;">Windfall Data Science Challenge</h1>

<div align="center">
  <img src="https://windfalldata.com/wp-content/uploads/2020/07/Windfall-Logo-348x120-1.png" alt="windfalldata" height="400"/>
  <br>
  <p id="desc" style="font-style:italic;text-align:center;">This project attempts to predict large donors for the Windfall Children's Center. </p>
</div>

## Contents
 [Donation Analysis + Cleaning](/Windfall_Analysis+Cleaning.ipynb) - This file cleans the donations.csv to have attributes that reflect a user's historic spending patterns.
 
 [Donor Predictions](/WindFall_Predictions.ipynb) - This file runs various classification algorithms to try and predict whether or not a user will donate $20k in the next 5 years.

 [Final Classification Model](/Windfall%20DS%20Challenge%20--%20PTG/finalModel.joblib) - This ML Model predicts whether or not a user will donate $20k in the next 5 years based on historic patterns.

 [Final Propensity Model](/Windfall%20DS%20Challenge%20--%20PTG/finalPropensityModel.joblib) - This ML Model returns the probability of a user donating $20k in the next 5 years based on historic patterns.

## Libaries Used
### Cleaning
* Numpy
* Pandas

### Visualization
* Matplotlib
* Seaborn

### Machine Learning
* Sklearn
* Xgboost

## Data Cleaning
### Attribute Engineering
* 'Age' was removed because this attribute proved to be accurate for less than 18% of the donors.
* The data was grouped by 'cand_id' and 'Year' to provide insight on historic spending behaviors.
* Time was incorporated into the attribute set via a user record's current 'Year', the user's 'Start Year', and the user's 'Years Spent' ('Year' - 'Start Year')
* Spending Trends were incorporated into the records via a 'Rolling Total' of all donations made and a 'Rolling Average' to reflect the average donation per yer.
* The target variable was adjusted to record the sum of the donations of a user's next 5 years (from a record's year). That number was then further normalized into 0 or 1 depending on whether or not the sum exceeded 20k.

### Data Removal
* User Records after 2014 were removed. This is because the target variable in records after 2014 incorrectly depicts donations made over less than 5 years.
* All rows in the fully merged dataset with atleast 1 null value were removed.  

### Feature Removal
* All input features were ranked by importance via RFE.
* The 5 attributes with the lowest importance were removed.

### Algorithm Selection
* Algorithn Selection was dictated by looking at F1 Score. The F1 score is the harmonic mean of the precision and recall.
* Overall Accuracy and AUC Score were also recorded.
* The algorithm with the best F1 Score was the Extra Trees Classifier.
* The Extra Trees Regressor was used to craft the probability (of $20k donated in the next 5 years) based propensity model. This works because the target variable is 1 or 0, and the Extra Trees Regressor uses the algorithmic framework of the Extra Trees Classifier with mean at the end (instead of mode).

### Hyperparameter Tuning
* A gridsearch was run on Bootstrap, Criterion, and Max Features

### Results
| Model | Description | Accuracy | F1 Score | AUC Score |
| ----------- | ----------- | ----------- | ----------- | ----------- |
| **Naïve** | Predicted 0 for Everything | 0.9806 | 0.0000 | 0.5000 |
| **Basic** | Extra Trees | 0.9899 | 0.7100 | 0.8186 |
| **Extended Attributes** | Basic + More Attributes  | 0.9936 | 0.8798 | 0.9238 |
| **Feature Optimized** | Extended Attributes - Unnecessary Attributes | 0.9936 | 0.8807 | 0.9249 |
| **Hyperparameter Optimized** | Feature Optimized + modified criterion | 0.9936 | 0.8815 | 0.9274 |

Final Model: **ExtraTreesClassifier with 23 Attributes and {bootstrap = False, criterion = 'entropy', max_features = 'sqrt'}**

## Usage Instructions
In order to create predictions with new data:
```python
from joblib import load
from sklearn.ensemble import ExtraTreesClassifier
model = load('Windfall DS Challenge -- PTG/finalModel.joblib')
model.predict(<Your Data>)
```
In order to find the probability of new data producing over $20k over the next 5 years:
```python
from joblib import load
from sklearn.ensemble import ExtraTreesRegressor
model = load('Windfall DS Challenge -- PTG/finalPropensityModel.joblib')
model.predict(<Your Data>)
```
* <Your Data> needs to be formatted like the input data used to build the final model.

### Future Work
* Removal of outlier donations (over 3.5 std deviations from the mean)
* Normalization of input variables (via Standard Scalar)
* More Hyperparameter tuning
* Advanced ML Algorithms (via Keras, TPOT)
