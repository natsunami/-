<h1>Project: Airbnb Price Prediction in Paris

![airbnb_paris.jpg](attachment:bf6f172f-1646-40d9-a34b-188285fef6aa.jpg)

Welcome to the **airbnb prediction price in Paris project** ! 

In this notebook we are going to predict the price of airbnb homes. To do this, In order to achieve this goal, we have carried out several steps:

- 1/ Data Preparation 
- 2/ Apply Pipelines
- 3/ ML regression Benchmarking
- 4/ Hypertuning
- 5/ Test on new data

The Dataset is composed of 66900 rows and 106 columns. Metrics used in this project were the **Mean Square Error (MSE)** and **R2 Score (r2)**.

You can get the paris dataset here : [Inside Airbnb](http://insideairbnb.com/get-the-data.html)

Also take a look at the notebooks produced by AWS SageMaker Autopilot:
- [sagemaker_autopilot_data_exploration_notebook.html](https://natsunami.github.io/Projects/Airbnb/sagemaker_autopilot_data_exploration_notebook.html)
- [Amazon SageMaker Autopilot Candidate Definition Notebook](https://natsunami.github.io/Projects/Airbnb/sagemaker_autopilot_candidate_notebook.html)

Time to code !

<h1> Import Packages


```python
import os, sys
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import xgboost as xgb 
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV, cross_validate
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, RobustScaler, OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA, TruncatedSVD
from category_encoders.target_encoder import TargetEncoder
from category_encoders import LeaveOneOutEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords

pd.options.mode.chained_assignment = None  # default='warn

```

<h1> Dataset Airnbnb Paris

First we open our csv and read it:


```python
path = '../../Datasets'
file_name = 'listings.csv'
filepath = os.path.join(path,file_name)

df = pd.read_csv(filepath)
```

    /home/natsunami/anaconda3/lib/python3.8/site-packages/IPython/core/interactiveshell.py:3145: DtypeWarning: Columns (43,61,62) have mixed types.Specify dtype option on import or set low_memory=False.
      has_raised = await self.run_ast_nodes(code_ast.body, cell_name,


We can now take a peak of what our data look like:


```python
print(f"The dataset is composed of {df.shape[0]} rows and {df.shape[1]} columns.")
df.head()
```

    The dataset is composed of 66900 rows and 106 columns.





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>listing_url</th>
      <th>scrape_id</th>
      <th>last_scraped</th>
      <th>name</th>
      <th>summary</th>
      <th>space</th>
      <th>description</th>
      <th>experiences_offered</th>
      <th>neighborhood_overview</th>
      <th>...</th>
      <th>instant_bookable</th>
      <th>is_business_travel_ready</th>
      <th>cancellation_policy</th>
      <th>require_guest_profile_picture</th>
      <th>require_guest_phone_verification</th>
      <th>calculated_host_listings_count</th>
      <th>calculated_host_listings_count_entire_homes</th>
      <th>calculated_host_listings_count_private_rooms</th>
      <th>calculated_host_listings_count_shared_rooms</th>
      <th>reviews_per_month</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2577</td>
      <td>https://www.airbnb.com/rooms/2577</td>
      <td>20200510041557</td>
      <td>2020-05-12</td>
      <td>Loft for 4 by Canal Saint Martin</td>
      <td>100 m2 loft (1100 sq feet) with high ceiling, ...</td>
      <td>The district has any service or shop you may d...</td>
      <td>100 m2 loft (1100 sq feet) with high ceiling, ...</td>
      <td>none</td>
      <td>NaN</td>
      <td>...</td>
      <td>t</td>
      <td>f</td>
      <td>strict_14_with_grace_period</td>
      <td>f</td>
      <td>f</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0.06</td>
    </tr>
    <tr>
      <th>1</th>
      <td>3109</td>
      <td>https://www.airbnb.com/rooms/3109</td>
      <td>20200510041557</td>
      <td>2020-05-13</td>
      <td>zen and calm</td>
      <td>Appartement très calme de 50M2 Utilisation de ...</td>
      <td>I bedroom appartment in Paris 14</td>
      <td>I bedroom appartment in Paris 14 Good restaura...</td>
      <td>none</td>
      <td>Good restaurants very close the Montparnasse S...</td>
      <td>...</td>
      <td>f</td>
      <td>f</td>
      <td>flexible</td>
      <td>f</td>
      <td>f</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0.22</td>
    </tr>
    <tr>
      <th>2</th>
      <td>5396</td>
      <td>https://www.airbnb.com/rooms/5396</td>
      <td>20200510041557</td>
      <td>2020-05-13</td>
      <td>Explore the heart of old Paris</td>
      <td>Cozy, well-appointed and graciously designed s...</td>
      <td>Small, well appointed studio apartment at the ...</td>
      <td>Cozy, well-appointed and graciously designed s...</td>
      <td>none</td>
      <td>You are within walking distance to the Louvre,...</td>
      <td>...</td>
      <td>t</td>
      <td>f</td>
      <td>strict_14_with_grace_period</td>
      <td>f</td>
      <td>f</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1.66</td>
    </tr>
    <tr>
      <th>3</th>
      <td>7397</td>
      <td>https://www.airbnb.com/rooms/7397</td>
      <td>20200510041557</td>
      <td>2020-05-13</td>
      <td>MARAIS - 2ROOMS APT - 2/4 PEOPLE</td>
      <td>VERY CONVENIENT, WITH THE BEST LOCATION !</td>
      <td>PLEASE ASK ME BEFORE TO MAKE A REQUEST !!! No ...</td>
      <td>VERY CONVENIENT, WITH THE BEST LOCATION ! PLEA...</td>
      <td>none</td>
      <td>NaN</td>
      <td>...</td>
      <td>f</td>
      <td>f</td>
      <td>moderate</td>
      <td>f</td>
      <td>f</td>
      <td>5</td>
      <td>5</td>
      <td>0</td>
      <td>0</td>
      <td>2.42</td>
    </tr>
    <tr>
      <th>4</th>
      <td>7964</td>
      <td>https://www.airbnb.com/rooms/7964</td>
      <td>20200510041557</td>
      <td>2020-05-12</td>
      <td>Large &amp; sunny flat with balcony !</td>
      <td>Very large &amp; nice apartment all for you!  - Su...</td>
      <td>hello ! We have a great 75 square meter apartm...</td>
      <td>Very large &amp; nice apartment all for you!  - Su...</td>
      <td>none</td>
      <td>NaN</td>
      <td>...</td>
      <td>f</td>
      <td>f</td>
      <td>strict_14_with_grace_period</td>
      <td>f</td>
      <td>f</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0.05</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 106 columns</p>
</div>



Okay, we can see that our DataFrame is quiet big...

We suppose that we won't need that much of features to predict the price so we will have to remove some columns, clean the remaining ones and process them.

<h1> Data Preparation

First, I checked for NaN values and applied a mask on my data to remove the columns were the number of NaN values were >= 30 % .


```python
isna_mask = (df.isna().sum()/df.shape[0]*100) < 30
df = df.loc[:, isna_mask]
```

I also cheked for duplicated rows based on the 'id' but nothing return.


```python
df_duplicated = df.duplicated(subset='id',keep=False)
df_duplicated[df_duplicated == True]
```




    Series([], dtype: bool)



After taking some time to inspect the data I remove a large number of columns (e.g: id, scraped_id, requiere_guest_profile_pic,..).

Of course I could have keep all the features and use an algorithm that is not affected by the curse of dimensionality, but the aim was to use features wich could bring significant information.


```python
columns_to_keep = [
 'host_since',
 'first_review',
 'last_review',
 'summary',
 'description',
 'zipcode',
 'property_type',
 'room_type',
 'accommodates',
 'bathrooms',
 'bedrooms',
 'beds',
 'bed_type',
 'amenities',
 'price',
 'security_deposit',
 'cleaning_fee',
 'guests_included',
 'extra_people',
 'minimum_nights',
 'maximum_nights',
 'number_of_reviews'
 ]

df = df.loc[:,columns_to_keep]

print(f"The dataset is now composed of {df.shape[0]} rows and {df.shape[1]} columns.")
```

    The dataset is now composed of 66900 rows and 22 columns.


Since i will be using some textual features ( 'summary' and ' description' which are quiet similar but gave the best results) and cannot fill NaN values with some new text i removed these rows


```python
df = df[~np.logical_or(df['summary'].isna(),df['description'].isna())]
```

I also separated my data according to dtype:


```python
df_numerical = df.select_dtypes(include=['int','float'])
df_others = df.select_dtypes(exclude=['int','float'])
```

A quick heatmap of correlation matrix of numerical data: 


```python
plt.figure(figsize=(9,7))
sns.heatmap(df_numerical.corr())
plt.title('Correlation Matrix plot of numerical data')

plt.show()
```


![png](output_23_0.png)


Based on the correlation matrix, I checked for higly correlated features (redundant information) using a mask (correlation > 0.8) but no features were found:


```python
corr_mask = np.triu(np.ones_like(df_numerical.corr().abs(), dtype = bool))
df_numerical_corr_masked = df_numerical.corr().abs().mask(corr_mask)

numerical_col_to_remove = [ c for c in df_numerical_corr_masked.columns if any(df_numerical_corr_masked[c] > 0.8)]

print(f'Higly correlated numerical features : {numerical_col_to_remove}')
```

    Higly correlated numerical features : []


I practiced feature engineering on 'minimum_nights' and 'maximum_nights', basically i just made the mean of the two features:


```python
df_numerical['mean_num_nights'] = (df_numerical['minimum_nights'] + df_numerical['maximum_nights'])/2
df_numerical.drop(['minimum_nights','maximum_nights'], axis=1, inplace=True)
```


```python

# Price_col cleaning:

priced_col = ['price','security_deposit','cleaning_fee','extra_people']

for col in priced_col:
    df_others[col] = df_others[col].str.replace('$','').str.replace(",","").astype('float')

# Amenities cleaning:

def amenities_cleaning(x):
    
    ''' This function is used to clean the Amenities column '''
    
    x =  len(x.replace('{','').replace('}','').split(','))
    return x

df_others['amenities'] = df_others['amenities'].apply(lambda x: amenities_cleaning(x))

# Zipcode cleaning ( zipcode values where value_counts was < 800 were replaced by 'Other'):

zipcode_mask = df_others['zipcode'].value_counts().index[df_others['zipcode'].value_counts() < 800]
mask_isin_zipcode = df_others['zipcode'].isin(zipcode_mask)
df_others['zipcode'][mask_isin_zipcode] = 'Other'
df_others['zipcode_clean'] = df_others['zipcode'].astype('str').apply(lambda x: x.replace(".0",""))
df_others.drop(['zipcode'], axis=1,inplace=True)

# Property_type cleaning ( property_type values where value_count was < 10 were replaced by 'Exotic'):

exotic_properties = df_others['property_type'].value_counts()[df_others['property_type'].value_counts() < 10].index
mask_isin_exotic = df_others['property_type'].isin(exotic_properties)
df_others['property_type'][mask_isin_exotic] = 'Exotic'

# Room_type and bed_type cleaning:
# No need since we don't have that much unique values :
print(f"Number of unique values for 'room_type' feature: {len(df_others['room_type'].unique())}")
print(f"Number of unique values for 'df_others' feature: {len(df_others['bed_type'].unique())}")
```

    Number of unique values for 'room_type' feature: 4
    Number of unique values for 'df_others' feature: 6


Finally we concat our two DataFrame after cleaning:


```python
df_final = pd.concat([df_others,df_numerical], axis=1)
```

Still before going to the next step, there's something quiet important that we need to check : **Price distribution**


```python
fig, axes = plt.subplots(1,2,figsize=(15,7))

sns.distplot(df_final['price'], ax=axes[0])
axes[0].set_title('Price distribution')

sns.distplot(df_final['price'][df_final['price'] < 500], ax=axes[1])
axes[1].set_title('Price distribution with label thresholding')

plt.show()
```


![png](output_32_0.png)


Okay, so as we can see, most of the prices are between 0 and 500, thus I've decided to removed the abnormal prices with a threshold at 500 and also applied a log transform. 

You may wonder **why a log transform** ? 

Well, it's just that without log transform on the target the algorithms performed quiet bad. I tested various transform and the logarithm transform gave us the best performances.


```python
df_final[df_final['price']< 500]
df_final['price_log'] = df_final['price'].apply(lambda x: np.log1p(x))
df_final.drop('price', axis=1, inplace=True)
```

Now, let's take a look at our target...The target feature is more Gaussian-Like now !


```python
fig, ax = plt.subplots()

sns.distplot(df_final['price_log'], ax=ax)
plt.title('Price with Log Transform')

plt.show()
```


![png](output_36_0.png)


If you wanna save the csv preprocessed: 


```python
#df_final.to_csv('airbnb_paris_df.csv',index=False)
```

We can load the preprocessed csv ready to use for ML purpose (need to remove the hashtag):


```python
#df_final = pd.read_csv('airbnb_paris_df.csv', index_col=False)
#df_final.drop('Unnamed: 0', axis=1, inplace=True)
```

 We split our DataFrame into a target columns y (price_log) and features columns X and perform the train-test-split:


```python
y = df_final['price_log']
X = df_final.drop(['price_log'], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X,y, shuffle=True, random_state= 42)
```

Just before doing ML I processed date  features  (host_since,  first_review,  last_review).
    
I converted  them  into continuous values by filling the null value with the mean date, and subtracting the earliest date value from all datevalues.


```python
date_columns = ['host_since','first_review','last_review']

for col in date_columns:
    X_train[col] = pd.to_datetime(X_train[col])
    X_test[col] = pd.to_datetime(X_test[col])
    mean_date = str(X_train[col].mean())
    X_train[col] = X_train[col].astype('str')
    X_test[col] = X_test[col].astype('str')
    imputer_date = SimpleImputer(missing_values='NaT',strategy='constant', fill_value=mean_date)
    X_train[col] = imputer_date.fit_transform(X_train[[col]])
    X_train[col] = pd.to_datetime(X_train[col])
    X_test[col] = imputer_date.transform(X_test[[col]])
    X_test[col] = pd.to_datetime(X_test[col])

for col in date_columns:
    X_train[col] = X_train[col] - X_train[col].min()
    X_train[col] = X_train[col].apply(lambda x: x.days)
    X_test[col] = X_test[col] - X_test[col].min()
    X_test[col] = X_test[col].apply(lambda x: x.days)
    
```

<h1> Processing Pipelines: 

Okay, we already made a lot of things such as cleaning and processing our data, Still we have more to do so that our preparation will be completed and we will use **Pipelines** to the rescue !

Pipelines allow us to perform various type of data preparation and chaining them such as:
- Imputing
- Scaling
- Encoding

Here is what i did :
- 1/ Impute Nan Values with the median and use a Standard Scaler on numerical data
- 2/ Impute Nan Values with the most frequent values and use a OneHotEncoder for categorical data
- 3/ Tf-idf with stop words in french and english with a truncatedSVD wich is a dimensionality reduction method for sparse matrix (a PCA won't work here).


```python

numerical_columns = ['host_since','first_review','last_review','amenities','security_deposit', 'cleaning_fee', 'extra_people','accommodates', 'bathrooms', 'bedrooms', 'beds', 'guests_included','number_of_reviews', 'mean_num_nights']
categorical_columns = ['property_type', 'room_type', 'bed_type','zipcode_clean']
text_columns = ['summary']
```


```python
final_stopwords_list = stopwords.words('english') + stopwords.words('french')

numerical_transformer = Pipeline(steps=[('imputer',SimpleImputer(strategy='median')),('scaler',StandardScaler())])

categorical_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='most_frequent')),('encoder',OneHotEncoder())])

text_transformer = Pipeline(steps=[('tfidf',TfidfVectorizer(stop_words=final_stopwords_list)),('svd', TruncatedSVD(n_components=50))])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_columns),
        ('cat', categorical_transformer, categorical_columns),
        ('text_summary', text_transformer, 'summary')])

#preprocessor_pca = Pipeline(steps=[('preprocess',preprocessor),('pca',T(n_components=0.90))])

```

Finally we can fit_transform on the train set and apply the transform on the test set which avoid data leaking ! (Thanks pipelines !)


```python
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)
```

<h1> Machine Learning

Finally, we can start to perform regression with our dataset prepared.

Because it is important to keep things simple at first, we will start with a linear regression.

<h3> Linear regression


```python
#Baseline model with linear regression:
lr = LinearRegression()

lr.fit(X_train_processed,y_train)

#Predict on train set:
y_pred_train_lr = lr.predict(X_train_processed)
mse_train_lr = mean_squared_error(y_train, y_pred_train_lr)
r2_train_lr = r2_score(y_train, y_pred_train_lr)

print(f'MSE Training Set: {mse_train_lr}')
print(f'R2 Training Set: {r2_train_lr}')

# Cross Validation 10 folds multi-scoring
lr_cv = cross_validate(lr,X_train_processed, y_train, scoring=['neg_mean_squared_error','r2'], cv=10, n_jobs=-1)
mse_val_lr = -lr_cv['test_neg_mean_squared_error'].mean()
r2_val_lr = lr_cv['test_r2'].mean()

print(f'MSE Validation Set: {mse_val_lr}')
print(f'R2 Score Validation Set: {r2_val_lr}')
```

    MSE Training Set: 0.18772402329657636
    R2 Training Set: 0.5700328472355666
    MSE Validation Set: 0.38188113627016496
    R2 Score Validation Set: 0.1276603896954363


Using a linear regression perform on our data gives us poor results. As we can see our model doesn't learn really well on the training set and therefore it does not manage to generalize well.

A suggestion would be to use non-linear models such as decision trees or random forests. 

<h3> Random Forest 


```python
#Baseline model with random forest regression:
rfr  = RandomForestRegressor()

rfr.fit(X_train_processed,y_train)

y_pred_train_rfr = rfr.predict(X_train_processed)
mse_train_rfr = mean_squared_error(y_train, y_pred_train_rfr)
r2_train_rfr = r2_score(y_train, y_pred_train_rfr)


print(f'MSE Training Set: {mse_train_rfr}')
print(f'R2 Training Set: {r2_train_rfr}')

# Cross Validation 10 folds multi-scoring
cv_rfr  = cross_validate(rfr,X_train_processed, y_train, scoring=['neg_mean_squared_error','r2'], cv=5, n_jobs=-1)

mse_val_rfr = -cv_rfr['test_neg_mean_squared_error'].mean()
r2_val_rfr = cv_rfr['test_r2'].mean()

print(f'MSE Validation Set: {mse_val_rfr}')
print(f'R2 Score Validation Set: {r2_val_rfr}')
```

    MSE Training Set: 0.02192841979170746
    R2 Training Set: 0.9497746742431146
    MSE Validation Set: 0.1604632027482325
    R2 Score Validation Set: 0.6324856043748844


Results demonstrated that using RF the model is really good  at learning the training set  but tends to overfit. 

Because the bootstrap aggregating technique also called "Bagging" can lead to mistakes if the majority of classifiers are wrong in the same regions and boosting models overcome this issue by training models sequentially.

Thus, we will finish our benchmark by using XGBoost which is an optimized distributed gradient boosting library. 

<h3> Gradient Boosting with XGBoost


```python
#Baseline model with XGBoost:
xboost = xgb.XGBRegressor(tree_method='gpu_hist', gpu_id=0) #Use of GPU

xboost.fit(X_train_processed, y_train)

y_pred_train_xgb = xboost.predict(X_train_processed)
mse_train_xgb = mean_squared_error(y_train, y_pred_train_xgb)
r2_train_xgb = r2_score(y_train, y_pred_train_xgb)

print(f'MSE Training Set: {mse_train_xgb}')
print(f'R2 Training Set: {r2_train_xgb}')
```

    MSE Training Set: 0.0768213271387374
    R2 Training Set: 0.8240467750403769



```python
xboost = xgb.XGBRegressor(tree_method='gpu_hist', gpu_id=0)
# Cross Validation 10 folds multi-scoring
xgb_cross_val = cross_validate(xboost, X_train_processed, y_train, scoring=['neg_mean_squared_error','r2'], cv=10)

mse_val_xgb = -xgb_cross_val['test_neg_mean_squared_error'].mean()
r2_val_xgb = xgb_cross_val['test_r2'].mean()

print(f'MSE Validation Set: {mse_val_xgb}')
print(f'R2 Score Validation Set: {r2_val_xgb}')
```

    MSE Validation Set: 0.14351182877820456
    R2 Score Validation Set: 0.6709652482285591


Yes ! We improved our metrics on the validation set, using a beseline gradient boosting algorithm with XGBoost disminished the overfit that we had using Random Forest.

<h1> Model Tuning

Comparing our baselines models we can clearly see that Gradient Boosting using XGBoost performed better on a validation set than a linear regression and a Random Forest.

Thus, tuning our boosting model could be quiet interesting in order to reduce the overfit. Nevertheless boosting models have a lot of hyperparameters (more than random forest)and tuning them is an hard task that take time and computational power.

Due to computational limitation on my computer, I switched to cloud computing but running the randomized search took me too much time. Implement the build-in XGBoost in SageMaker would be preferable (see [xgboost in SageMaker](https://docs.aws.amazon.com/sagemaker/latest/dg/xgboost.html)).

I decided to use the hyperparameters given by the best model of SageMaker Autopilot:
![hyp_tun_sage.png](attachment:2ed7fcb2-4af4-407f-987d-94aee37f60f5.png)



```python
#Example of XGBoost tuning that can be made:
#params_xgb = {'n_estimators':range(100,1000,20),
#         'max_depth':range(1,30),
#         'learning_rate':np.linspace(0.001,0.3,num=50),
#         'reg_alpha':np.linspace(0.001,0.5,num=50)}

#rand_search_xgb = RandomizedSearchCV(xboost,params_xgb,n_iter=60, cv=5)
#rand_search_xgb_results = rand_search_xgb.fit(X_train_processed, y_train)
#rand_search_xgb_results.best_params_
#rand_search_xgb_results.best_score_
```


```python
#Hyperparameters were calculated by SageMaker Autpilot giving the best model:

xboost = xgb.XGBRegressor(n_estimators=740,
                          max_depth= 6, 
                          learning_rate=0.11214298095296928,
                          reg_alpha=0.317940445687246,
                          colsample_bytree= 0.38523465777592514,
                          gamma=4.465918775019842e-06,
                          min_child_weight = 0.8883356598036687,
                          subsample = 0.86520980408866,
                          tree_method='gpu_hist', gpu_id=0)

# Cross Validation 10 folds multi-scoring
xgb_cross_val = cross_validate(xboost, X_train_processed, y_train, scoring=['neg_mean_squared_error','r2'], cv=10)

mse_val_xgb = -xgb_cross_val['test_neg_mean_squared_error'].mean()
r2_val_xgb = xgb_cross_val['test_r2'].mean()

print(f'MSE Validation Set: {mse_val_xgb}')
print(f'R2 Score Validation Set: {r2_val_xgb}')
```

    MSE Validation Set: 0.13476272715926674
    R2 Score Validation Set: 0.6910266578948767


That's quiet impressive ! We improved again our model using SageMaker Autopilot hyperparameters.

Now it is time to try our model on new data !

<h1> Predictions on new data

We fit our XGBoost model on the training set and predict on the test set:


```python
xboost = xgb.XGBRegressor(n_estimators=740,
                          max_depth= 6, 
                          learning_rate=0.11214298095296928,
                          reg_alpha=0.317940445687246,
                          colsample_bytree= 0.38523465777592514,
                          gamma=4.465918775019842e-06,
                          min_child_weight = 0.8883356598036687,
                          subsample = 0.86520980408866)

xboost.fit(X_train_processed, y_train)

y_pred_xgb = xboost.predict(X_test_processed)
mse_xgb_test = mean_squared_error(y_test, y_pred_xgb)
r2_xgb_test = r2_score(y_test, y_pred_xgb)

print(f'MSE Test Set: {mse_xgb_test}')
print(f'R2 Score test Set: {r2_xgb_test}')
```

    MSE Test Set: 0.1407602089592525
    R2 Score test Set: 0.6761001859077365


We reconstruct our train set and apply an exponential transformation on the price_log and  predicted price (which is also logarithm tranqformed).


```python
test_set = pd.concat([X_test,y_test], axis=1)
test_set['price'] = np.expm1(test_set['price_log'])
test_set['price_predict'] = np.expm1(y_pred_xgb)
```

Let's take a look at our Airbnb predictions...


```python
test_set.head(10)[['summary','price','price_predict']]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>summary</th>
      <th>price</th>
      <th>price_predict</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>13377</th>
      <td>Appartement moderne très lumineux et bien agen...</td>
      <td>125.0</td>
      <td>89.331535</td>
    </tr>
    <tr>
      <th>55546</th>
      <td>Grand studio (31 m²) avec terrasse privative (...</td>
      <td>60.0</td>
      <td>50.031090</td>
    </tr>
    <tr>
      <th>13203</th>
      <td>Joli appartement 2 pièces, 40 m2, à 5 mn à pie...</td>
      <td>60.0</td>
      <td>62.725811</td>
    </tr>
    <tr>
      <th>35560</th>
      <td>Central Paris, 2 bedrooms, up to 6 people. AC ...</td>
      <td>237.0</td>
      <td>271.685272</td>
    </tr>
    <tr>
      <th>48834</th>
      <td>Studio lumineux avec cuisine équipée à 5 minut...</td>
      <td>75.0</td>
      <td>61.277748</td>
    </tr>
    <tr>
      <th>53791</th>
      <td>A 5 minutes du métro dans un quartier animé pr...</td>
      <td>80.0</td>
      <td>78.184563</td>
    </tr>
    <tr>
      <th>59203</th>
      <td>Charmant appartement de 20m2 situé dans une ru...</td>
      <td>85.0</td>
      <td>78.691383</td>
    </tr>
    <tr>
      <th>59326</th>
      <td>Charmant studio sous les toits avec cheminée d...</td>
      <td>75.0</td>
      <td>70.596603</td>
    </tr>
    <tr>
      <th>58524</th>
      <td>C'est super</td>
      <td>98.0</td>
      <td>110.149590</td>
    </tr>
    <tr>
      <th>20346</th>
      <td>Mon logement est proche de Nation. il est parf...</td>
      <td>45.0</td>
      <td>51.753811</td>
    </tr>
  </tbody>
</table>
</div>




![rsz_1rsz_octocat1.png](attachment:391a978a-92eb-400f-8c64-b0682fe18342.png)

Sounds good! 

We finally made it, this is the end of the **Airbnb prediction price** , I really hope that you find that interesting.

Also,  go check this scientific paper if you are interested in the subject : [Predicting Airbnb Listing Price Across Different Cities](http://cs229.stanford.edu/proj2019aut/data/assignment_308832_raw/26647491.pdf)
