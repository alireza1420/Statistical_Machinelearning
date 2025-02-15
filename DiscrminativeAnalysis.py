#!/usr/bin/env python
# coding: utf-8

# In[65]:


import random
import random
import numpy as np
import pandas as pd
from dateutil.rrule import weekday
import seaborn as sns
from scipy.linalg import solve
from scipy.ndimage import label
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform
from sklearn.model_selection import cross_val_score



# <h1>Discriminant analysis: LDA, QDA </h1>
# <h2> abstraction of the problem</h2>

# A classification problem aims to model the relationship between an input x and a categorical output y. In this project, the Capital Bikeshare bikes is trying to understand if an increase in number of currently available bikes is necessary or not.

# <h1>Generative models</h1>
# <p>A generative model, shows how an input X and an outpu Y is generated via P(y|x)</p>
# $$P(A \mid B) = \frac{ P(B \mid A) P(A) }{ P(B) }$$
#  A generative model calssifier is based on the following conditional probability, where y=m occures if X is met in Baye's theorem:
#  $$P(y=m \mid x) = \frac{ P(y=m) P(x\mid y=m) }{ \sum P(x\mid m) P(m) }$$
# 
#  
# 

# <h1>Normalization</h1>
# <p>LDA is a generative classifier model assuming the data is normaly distributed. This requires our data to be normalized since many of the features, including(Temp, Dew, Humidity, Windspeed, Cloudcover, Visibility) have a wide range of values. In addition, due to lack of variability and meaningfullness, we droped the Snow and Snow Depth.</p> We have chosen to procceed with two methods</p>
# 
# <ul>
# <li> Z-score
# <li> Min-Max
# </ul>
# <h1> Z-score </h1>
# <p> The standard deviation shows how much the data deviates from the mean. a small standar deviation makes the data less spread out around the mean.</p>

# LDA model assume all data follow the gausain distribution, meaning they don't varry from eachother. In addition, LDA, unlike QDA, assumes features have the same mean. Due to these requirments, in the event we notice our data's variance is too large, we need to perfrom standarzation.

# In[ ]:





# In[66]:


df=pd.read_csv('training_data_vt2025.csv')
df


# In[ ]:





# <h1>Model condition before </h1>

# In[67]:


np.random.seed(0)
mcb_trainI=np.random.choice(df['month'].shape[0],size=1200,replace=False)
index=df.index.isin(mcb_trainI)
mcb_train=df.iloc[index]
mcb_test=df.iloc[~index]
mcb_train.shape[0]
mcb_train_x=mcb_train.drop(columns=['increase_stock'])
mcb_test_x=mcb_test.drop(columns=['increase_stock'])
mcb_train_y=mcb_train['increase_stock']
mcb_test_y=mcb_test['increase_stock']
mcb_test_x


# In[68]:


model = LinearDiscriminantAnalysis(solver='lsqr', shrinkage="auto")

model.fit(mcb_train_x, mcb_train_y)
predictions = model.predict(mcb_train_x)
model.predict(mcb_train_x)
cm = pd.crosstab(mcb_train_y, model.predict(mcb_train_x))

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()
accuracy = accuracy_score(mcb_train_y, predictions)
accuracy 


# In[69]:


scores = cross_val_score(model, mcb_train_x, mcb_train_y, cv=5, scoring='accuracy')

print("Stratified k-Fold scores:", scores)


# In[ ]:





# <h1>Model Tuning </h1>
# <p> A machine learning model learns from training exisitng data to fir the model parameters. However there are other parameters involved in trainign a model called, Hyperparameters. They are external configuraiton variables that are used to manage models, like how fast it should learn or its complexity. Hyperparameter tuning helps a model improve its accuracy and prevents overfitting which can be beneficial for LDA. We have used two Hyperparameter Tuning techniques in this analysis </p>
# <ul>
# <li>GridSearch: Tests all possible hyperparameters to find the optimal one, However it is computationaly expensive but accurate.
# <li>RandomizeSearch:Randomly sample hyperparameter to find an optimal combination. Faster but it is possible to miss a combination
# </ul>

# In[70]:


# all 1600 data
tcp=df.drop(['increase_stock'],axis=1)
tcp_y=df['increase_stock']
tcp_y


# In[71]:


#GridSearch
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import GridSearchCV

# Define LDA and parameter grid
lda = LinearDiscriminantAnalysis()
param_grid = {
    'solver': [ 'lsqr','eigen'],
    'shrinkage': [None, 'auto', 0.1, 0.5, 0.9]  # Only used for 'lsqr' and 'eigen'
}

# Run GridSearchCV
grid_search = GridSearchCV(lda, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(tcp, tcp_y)

# Print best parameters and best score
print("Best Parameters:", grid_search.best_params_)
print("Best Accuracy:", grid_search.best_score_)


# In[72]:


#RandomSearch


# Define LDA model
lda = LinearDiscriminantAnalysis()

# Define correct hyperparameter distribution
param_dist = {
    "solver": ['lsqr', 'eigen'],  # 'svd' does not support shrinkage
    "shrinkage": uniform(0, 1),  # Random float between 0 and 1
}

# Run RandomizedSearchCV
lda_cv = RandomizedSearchCV(lda, param_distributions=param_dist, n_iter=10, cv=5, scoring='accuracy', random_state=42, n_jobs=-1)
lda_cv.fit(tcp, tcp_y)

# Print best parameters and score
print("Tuned LDA Parameters: {}".format(lda_cv.best_params_))
print("Best score is {}".format(lda_cv.best_score_))


# In[73]:


from sklearn.metrics import roc_curve, auc
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import matplotlib.pyplot as plt

# Train LDA model
lda = LinearDiscriminantAnalysis(solver='lsqr', shrinkage="auto")
lda.fit(tcp, tcp_y)  # Make sure X_train and y_train are defined

# Check class labels
print("Classes:", lda.classes_)

# Find the index for positive class
pos_label = 'high_bike_demand'  # Ensure this matches the actual label name
pos_index = list(lda.classes_).index(pos_label)

# Predict probabilities for the positive class
y_prob = lda.predict_proba(tcp)[:, pos_index]  # Ensure X_test is defined

# Compute ROC curve
fpr, tpr, thresholds = roc_curve(tcp_y, y_prob, pos_label=pos_label)
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f"LDA ROC curve (area = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')  # Diagonal line
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve for LDA Model")
plt.legend(loc="lower right")
plt.grid()
plt.show()


# In[74]:


scores = cross_val_score(model, tcp, tcp_y, cv=5, scoring='accuracy')

print("Stratified k-Fold scores:", scores)



# <h1>Feature Selection</h1>
# <p> 
# Feature selection is the process of reducing the number of input variables in a model by retaining only the most relevant features. By eliminating noise in the data, it can improve the model's accuracy and efficiency.</p>
# <p>
# Covariance is a metric that measures the linear relationship between variables. Since LDA assumes that the data follows a normal distribution with equal class covariances, selecting features that align with this assumption can enhance its performance.
# </p>
# 

# In[75]:


# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(tcp, tcp_y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train LDA model
lda = LinearDiscriminantAnalysis()
lda.fit(X_train_scaled, y_train)

# Get feature importance (absolute values of LDA coefficients)
feature_importance = np.abs(lda.coef_[0])  # LDA coefficients
feature_names = tcp.columns  # Feature names

# Create DataFrame for plotting
importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importance})
importance_df = importance_df.sort_values(by='Importance', ascending=False)

# Plot feature importance
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=importance_df, palette='viridis')
plt.xlabel('Coefficient Magnitude (Importance)')
plt.ylabel('Feature Name')
plt.title('Feature Importance in LDA')
plt.show()


# <h1>Seperating Varibales into train and test</h1>

# In[76]:


df['month'].shape[0]


# In[77]:


np.random.seed(0)
trainI=np.random.choice(df['month'].shape[0],size=800,replace=False)
index=df.index.isin(trainI)
train=df.iloc[index]
test=df.iloc[~index]
train.shape[0]
train


# <h1>Independent and Dependent Variables</h1>
# 
# 

# In[78]:


features = ['hour_of_day','month','weekday','temp', 'dew', 'humidity', 'holiday', 'cloudcover']

X_train=train[features]
y_train=train['increase_stock']
X_test=test[features]
y_test=test['increase_stock']
#Before standarzation
X_test.describe()


# <h1>Train and Test Sets</h1>
# 
# The X dataset has our ten features and the y dataset has our dependent variable which is increase stock or decrease stock. We
# can not split our data into a train and test set. The code is below.
# 
# The training data set consist of 70% and the test data set is consist of 30% of our actual data. 

# In[79]:


X_train.describe()


# In[80]:


scaler = StandardScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.fit_transform(X_test)


# In[81]:


X_train=pd.DataFrame(X_train)
X_test=pd.DataFrame(X_test)


# In[82]:


X_train.describe()


# <h1>Training On given data</h1>

# ### Formula(LDA)
# To calculate LDA, we will be using the logic behind beyers theorom, where we need Two parameters:
# 1- Prior class probabilities  
# 2- The conditional probability density 
# 
# 
# To implement the LDA, we need to know the bayes rule. And it requires to implement probability distribution function. Basic notation is like this:
# $$
# g_m(x) = \frac{\pi_m f_m(x)}{\sum_{m=1}^{M} \pi_m f_m(x)}
# $$
# $${\pi_m =\frac{ n_m}{n}}$$
# 
# 
# $$ f_m(x) = \frac{1}{\vert 2 \pi \Sigma \vert^{\frac{1}{2}}} \text{exp} \big( -\frac{1}{2} (x - \mu)^T (\Sigma)^{-1} (x - \mu)\big) $$
# 
# 
# In our case, we either have the increase or decrease so our m is either 0 or 1, which by putting them in the formula mentioned above, we reach the following probability distribution:
# 
# $$ f(x \vert y=0) = \frac{1}{\vert 2 \pi \Sigma^{(0)} \vert^{\frac{1}{2}}} \text{exp} \big( -\frac{1}{2} (x - \mu^{(0)})^T (\Sigma^{(0)})^{-1} (x - \mu^{(0)})\big)\\
# $$
# 
# $$f(x \vert y=1) = \frac{1}{\vert 2 \pi \Sigma^{(1)} \vert^{\frac{1}{2}}} \text{exp} \big( -\frac{1}{2} (x - \mu^{(1)})^T (\Sigma^{(1)})^{-1} (x - \mu^{(1)})\big) $$
# 
# The model was build to maximize multivariate guassian distribution, meaning if we have multiple features, it chooses the best guasian distribution according to them, However, with one difference :
# 
# 1- LDA assumes all features have the samve covariance, their measurements might be different but they are all centered around the same area
# 
# 2- QDA shows no assumption between datas, making it a more realistic but complexer model
# 
# Maximizing the mentioned formula will result in

# In[83]:


model=LinearDiscriminantAnalysis(solver='lsqr',shrinkage="auto") 

model.fit(X_train,y_train)
predictions=model.predict(X_test)
model.predict(X_test)
cm=pd.crosstab(y_train,model.predict(X_train))

plt.figure(figsize=(10,8))
sns.heatmap(cm,annot=True)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()


# In[84]:


scores = cross_val_score(model, X_train,y_train, cv=5, scoring='accuracy')

print("Stratified k-Fold scores:", scores)


# In[85]:


import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Get predicted probabilities
y_prob = model.predict_proba(X_train)

# Convert to a DataFrame for easier plotting
df = pd.DataFrame(y_prob, columns=model.classes_)

# Plot density distribution of predicted probabilities
plt.figure(figsize=(8, 5))
for c in model.classes_:
    sns.kdeplot(df[c], label=f'Class {c}', fill=True)

plt.xlabel('Predicted Probability')
plt.ylabel('Density')
plt.title('Class Probability Distributions')
plt.legend()
plt.show()


# In[86]:


print('Accuracy:',accuracy_score(y_test, model.predict(X_test)))
X_train


# In[87]:


from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

# Assuming your classifier is 'lda' and you've trained it already
# Ensure that you extract the probability for the positive class.
# Check the order of classes:
print("Classes:", model.classes_)

# Find the index corresponding to 'high_bike_demand'
pos_index = list(model.classes_).index('high_bike_demand')
y_prob = model.predict_proba(X_train)[:, pos_index]

# Compute ROC curve, explicitly setting pos_label
fpr, tpr, thresholds = roc_curve(y_train, y_prob, pos_label='high_bike_demand')
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f"LDA ROC curve (area = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')  # Diagonal line
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve for LDA Model")
plt.legend(loc="lower right")
plt.grid()
plt.show()


# In[88]:


from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

# Assuming your classifier is 'lda' and you've trained it already
# Ensure that you extract the probability for the positive class.
# Check the order of classes:
print("Classes:", model.classes_)

# Find the index corresponding to 'high_bike_demand'
pos_index = list(model.classes_).index('high_bike_demand')
y_prob = model.predict_proba(X_test)[:, pos_index]

# Compute ROC curve, explicitly setting pos_label
fpr, tpr, thresholds = roc_curve(y_test, y_prob, pos_label='high_bike_demand')
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f"LDA ROC curve (area = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')  # Diagonal line
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve for LDA Model")
plt.legend(loc="lower right")
plt.grid()
plt.show()


# In[89]:


train.columns


# In[93]:


scores = cross_val_score(model, X_test, y_test, cv=5, scoring='accuracy')

print("Stratified k-Fold scores:", scores)


# <h1>QDA</h1>

# In[90]:


model=QuadraticDiscriminantAnalysis()
model.fit(X_train,y_train)
predictions=model.predict(X_test)
model.predict(X_test)
cm=pd.crosstab(y_train,model.predict(X_train))

plt.figure(figsize=(10,8))
sns.heatmap(cm,annot=True)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()


# In[91]:


predictions = model.predict(X_test)

correct_predictions=np.sum(y_test==model.predict(X_test))
accuracy=correct_predictions/len(y_test)
print(f"Accuracy: {accuracy}")


# In[92]:


from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

# Assuming your classifier is 'lda' and you've trained it already
# Ensure that you extract the probability for the positive class.
# Check the order of classes:
print("Classes:", model.classes_)

# Find the index corresponding to 'high_bike_demand'
pos_index = list(model.classes_).index('high_bike_demand')
y_prob = model.predict_proba(X_test)[:, pos_index]

# Compute ROC curve, explicitly setting pos_label
fpr, tpr, thresholds = roc_curve(y_test, y_prob, pos_label='high_bike_demand')
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f"LDA ROC curve (area = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')  # Diagonal line
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve for QDA Model")
plt.legend(loc="lower right")
plt.grid()
plt.show()

