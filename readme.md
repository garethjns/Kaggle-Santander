# Kaggle Santander
Data and description: [https://www.kaggle.com/c/santander-customer-satisfaction][1]

Objective: Work out which customers are unhappy

Position: 1817/5123 (top 36%)

## Methods
 - Boosted trees (R - main6.R)
	* Feature engineering:
		* Error code counts
		* Identification of various vars based on forum exploration 
	* Preprocessing
		* Removal of numeric error codes
		* Removal or zero variance features (caret)
		* Removal of identical features
		* PCA (caret)
	* Model training:
		* Boosted trees (xgboost)
	* Postprocessing
		* Apply general "happy rules" - ideas from forums, ultimately leads to overfitting


 - ExtraTrees and random forests (Python - main.py)
	* Feature engineering/Preprocessing:
		* None!
	* Model training
		* Extra trees (sklearn)
		* Random forest classifier (sklearn)
		* maxDepth and nEstimators parameter grid search


 - ExtraTrees and random forests (Python - main5.py)
	* Feature engineering:
		* Error code counts
	* Preprocessing
		* Removal of numeric error codes
		* Removal or zero variance features (based on standard deviation)
	* Postprocessing
		* Apply general "happy rules"


[1]: https://www.kaggle.com/c/santander-customer-satisfaction