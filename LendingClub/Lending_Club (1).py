
# coding: utf-8

# # Lending Club Loan Dataset # 
# 
# **Task: predict the interest rate and grade assigned to a loan from other features.**

# **Preprocessing Data and Exploratory Data Analysis**

# In[346]:


#importing neccessary modules
import pandas as pd
import numpy as np
import tensorflow as tf 

from IPython.display import clear_output

#sci kit learn preprocessing and predictive model modules
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.svm import SVR
from sklearn.svm import LinearSVC
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.utils import shuffle
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import GradientBoostingRegressor

#imort datetime for datetime calculations
from datetime import datetime


# In[300]:


#loading data into memory of rejected 
#and accepted loan applicants 
pd.set_option('display.max_columns', 500)
data_path = "Downloads/LoanStats3a (1)_edited.csv"
df_orig = pd.read_csv(data_path,low_memory = False)

data_path_rejected = "Downloads/RejectStatsA (1).csv"
df_orig_rejected = pd.read_csv(data_path_rejected,low_memory = False)


# In[301]:


#having a look at data from both sets
df_orig_orig.head()


# In[302]:


df_orig_rejected.head()


# In[303]:


#some functions for initial data processing and feature engineering
def calc_months(month): 
    """
    calculates the number 
    of months from a given date
    """
    try: 
        month = str(month)
        month = month.replace("-", " ")
        month_list = month.split()
        month_number = datetime.strptime(month_list[0], '%b').month
        total_months = (12*(2011-int((month_list[1]))+1))-month_number
        return total_months
    except: 
        pass
    
def turn_NaN_to_mean(df):    
    """
    assigns NaN values to 
    the mean of that column
    """
    df = df.apply(lambda x: x.fillna(x.mean()),axis=0)
    return df

def convert_home_ownership(type_of_home): 
    """
    Numerical representation
    of home ownership status
    """
    #assiging different numerical values 
    #to different strings
    type_of_home = str(type_of_home)
    if type_of_home == "RENT": 
        type_of_home = 1
    elif type_of_home == "MORTGAGE": 
        type_of_home = 2
    elif type_of_home == "OWN": 
        type_of_home = 3
    return type_of_home

def convert_verification(type_of_verifcation): 
    """
    Numerical representation of 
    of verification status 
    """
    type_of_verifcation = str(type_of_verifcation)
    #assiging different numerical values 
    #to different strings 
    if type_of_verifcation == "Not Verified": 
        type_of_verifcation = 1
    elif type_of_verifcation == "Source Verified": 
        type_of_verifcation = 2
    elif type_of_verifcation == "Verified": 
        type_of_verifcation = 3
    return type_of_verifcation

def delete_NaN_columns(df):
    """
    deletes columns with 
    a large amount of NaN 
    values 
    """
    #dropping data columns with more 
    #than 25% missing values 
    df = df.dropna(thresh=len(df)/4, axis=1)
    return df 
    
def remove_percent(string): 
    """ 
    removes last character 
    from string in order to 
    get rid of percent signs
    """
    string = string[:-1]
    return stiring
    
def get_rid_of_NONE(inpt): 
    if inpt == 'NONE': 
        out = 0 
    return out

def strip_space(string): 
    """
    function to strip
    space from strings
    """
    string = string.strip() 
    return string 


def encode_categorical(df, column): 
    """ 
    encodes categorical 
    parameters as numerical values
    """
    #loading labeller
    labeller = preprocessing.LabelEncoder()
    #assiging data as input column and converting to list
    categorical_data = df[column]
    categorical_data = (np.array(categorical_data)).tolist()
    #fitting labeller to the categorical data
    labeller.fit(categorical_data)
    #applying labeller
    data_encoded = labeller.transform(categorical_data)
    df[column] = data_encoded
    return df    


# In[304]:


#stripping spaces and deleting percent signs from dti column 
dti_rej = df_orig_rejected['Debt-To-Income Ratio']
dti_rej = (np.array(dti_rej)).tolist()
for i in range(len(dti_rej)): 
    try: 
        dti_rej[i] = strip_space(dti_rej[i])
        dti_rej[i] = dti_rej[i][:-1]
        dti_rej[i] = float(dti_rej[i])
    except: 
        dti_rej[i] = ""


# In[305]:


#deleting non-numerical columns from various columns
df_orig_rejected['Debt-To-Income Ratio'] = dti_rej
df_orig_rejected['Employment Length'] = df_orig_rejected['Employment Length'].str.extract('(\d+)', expand=False)
df_orig['emp_length'] = df_orig['emp_length'].str.extract('(\d+)', expand=False)


# In[306]:


df_orig.head()


# In[307]:


#assiging grade labels for later training 
grade_targets = df_orig['grade']


# In[308]:


#Dropping unneeded columns for now (going to try and predict 
#interest rate)
df = df_orig
df.drop(['grade', 'sub_grade', 'emp_title'],axis=1, inplace=True)


# In[309]:


#getting rid nuisance characters in otherwise numerical  
#columns
df['term'] = df['term'].str.extract('(\d+)', expand=False)
df['emp_length'] = df['emp_length'].str.extract('(\d+)', expand=False)
df['revol_util'] = df['revol_util'].str.extract('(\d+)', expand=False)


# In[310]:


#getting rid of percent signs and converting to float 
#for interest rate column
int_rates = df['int_rate']
int_rates = (np.array(int_rates)).tolist()
for i in range(len(int_rates)-10): 
    try: 
        int_rates[i] = strip_space(int_rates[i])
        int_rates[i] = int_rates[i][:-1]
        int_rates[i] = float(int_rates[i])
    except: 
        del int_rates[i]

df = df[:-2]
df['int_rate'] = int_rates


# In[311]:


#converting dates to months since the date given 
df['last_credit_pull_d'] = df['last_credit_pull_d'].apply(calc_months)
df['settlement_date'] = df['settlement_date'].apply(calc_months)
df['debt_settlement_flag_date'] = df['debt_settlement_flag_date'].apply(calc_months)
df['last_pymnt_d'] = df['last_pymnt_d'].apply(calc_months)
df['earliest_cr_line'] = df['earliest_cr_line'].apply(calc_months)
df['issue_d'] = df['issue_d'].apply(calc_months)


# In[312]:


#filling NaN with the mean for various columns
df['last_credit_pull_d'].fillna((df['last_credit_pull_d'].mean()), inplace=True)
df['last_pymnt_d'].fillna((df['last_pymnt_d'].mean()), inplace=True)
df['earliest_cr_line'].fillna((df['earliest_cr_line'].mean()), inplace=True)
df['issue_d'].fillna((df['issue_d'].mean()), inplace=True)


# In[313]:


#converting home ownership and verification status with functions 
#defined earlier 
df['home_ownership'] = df['home_ownership'].apply(convert_home_ownership)
df['verification_status'] = df['verification_status'].apply(convert_verification)


# In[314]:


#deleting columns that seem to be useless
del df['pymnt_plan']
del df['title']
del df['initial_list_status']
del df['application_type']
del df['hardship_flag']
del df['disbursement_method']
del df['debt_settlement_flag']
del df['collections_12_mths_ex_med']
del df['delinq_amnt']
del df['pub_rec_bankruptcies']
del df['tax_liens']


# In[315]:


#encoding various columns categorically and attempting to 
#convert all cells to numerical values 
df = encode_categorical(df, "purpose")
df = encode_categorical(df, "zip_code")
df = encode_categorical(df, "loan_status")
df = encode_categorical(df, "addr_state")
df = df.convert_objects(convert_numeric=True)


# In[316]:


#getting rid of NaN heavy columns
df = delete_NaN_columns(df)
#turning NaN cells to mean of that column 
df = turn_NaN_to_mean(df)
#getting column list and looking at data 
df_for_corr = df
df_columns_list = df_for_corr.columns.tolist()
df.head()


# In[317]:


#Standerdizing data by centering to the mean and 
#calculating pearson correlation  
df_scaled = preprocessing.scale(df_for_corr) 
df_p = pd.DataFrame(df_scaled)
correlation = df_p.corr() 
correlation.columns = df_columns_list


# I am going to look for any obvious correlations between two parameters that could be used to build a simple model.

# In[318]:


#calculating correlations with interest rate parameter
correlation.iloc[[4]]


# There are no strong correlations between one parameter and interest rate. Due to the availability of machine learning models that can be implemented quickly, I am going see if any interesting results can be produced via a supervised machine learning regression or classification model. In these initial prediction models, I am going to aim to predict interest rates. In the next phase I will look at different classification algorithms for Loan grade. There is a well documented relationship between loan grade and interest rate. Therefore, if you can predict one you can predict the other with a high amount of certainty. Before I build these models I need to decide if any relevant data can be extracted from rejected loan data.

# # What to do about rejected loan data? #

# After looking initially at the different parameters for rejected loan data I can see there a few parameters which are shared with accepted loan data. I initially thought perhaps a very high interest rate could be assigned to the rejected data and a sample could be concated with the accepted loan data. Intuitively, this would train a model to look out for parameters (eg. high DTI ratio) that would be red flags and produce high interest rates.

# I am now going to preprocess a few parameters from both datasets so that sklearn's ".describe()" function can work on them.

# In[319]:


df['dti'].describe()


# In[320]:


df_orig_rejected['Debt-To-Income Ratio'].describe()


# In[321]:


df_orig_rejected['Amount Requested'].describe()


# In[322]:


df['loan_amnt'].describe()


# I am not going to include rejected loan data in my model. It is clear from summary statistics of the standard deviation and the mean, as well as min and max values, that the rejected data can not be considered to be a class of loans. There is too much variance to assign one interest rate or loan class to the rejected data. For example, the standard deviation of loan amount representing seven different loan classes is smaller than the standard deviation representing the "rejected" loan class.
# 
# There seems to be outliers or "maximum" debt to income ratios. I don't know how the data was collected and so don't know if these extremely high values are valid and so wouldn't know how to process this parameter. 
# 
# Applying any sort of unsupervised algorithm to assign interests rates to the unlabelled data is beyond the scope of this excercise.

# # Regression Techniques for Interest Rate Prediction #

# Due to the nature of the problem of predicting interest rates (continous value to predict) regression models are most suitable. I am going to use k-means cross validation to validate my models as it is easy to implement and provides robust validation. 
# 
# I am going to use R-squared values and MSE to evaluate the sucess of the various regression models.

# In[323]:


df = turn_NaN_to_mean(df)


# In[324]:


#shuffling data set and defining training and labels
df=shuffle(df)
df_target = df['int_rate']
df_train = df.drop(['int_rate'],axis=1)
train = (np.array(df_train))
labels = (np.array(df_target))


# **Preliminary testing of models** 

# Preliminary testing is done without validation to save time.

# In[330]:


#making training and test data sets
X_train, X_test = train[0:34031], train[34032:42539]
y_train, y_test = labels[0:34031], labels[34032:42539]
    
#Support Vector Regression 
SVR_model = SVR(C=1.0, epsilon=0.2)
SVR_model.fit(X_train, y_train)
SVR_pred = SVR_model.predict(X_test)

print("Support Vector Regression R^2 value: "+str(r2_score(y_test, SVR_pred)))
print("Support Vector Regression MSE value: "+str(mean_squared_error(y_test, SVR_pred)))


# In[349]:


#Gradient Boosting Regressor
GBR = GradientBoostingRegressor()
GBR.fit(X_train, y_train)
y_pred_RF = GBR.predict(X_test)
print("Gradient Boosting Regression R^2 value: "+str((r2_score(y_test, y_pred_RF))))
print("Gradient Boosting Regression MSE value: "+str(mean_squared_error(y_test, y_pred_RF)))


# In[334]:


#Random Forest 
RF = RandomForestRegressor(n_estimators=20, random_state=36)
RF.fit(X_train, y_train)
y_pred_RF = RF.predict(X_test)
print("Random Forest Regression R^2 value: "+str((r2_score(y_test, y_pred_RF))))
print("Random Forest Regression MSE value: "+str(mean_squared_error(y_test, y_pred_RF)))


# In[337]:


#Ada boost regressor
ADA = AdaBoostRegressor() 
ADA.fit(X_train, y_train)
y_pred_ADA = ADA.predict(X_test)
print("Ada Boost Regression R^2 value: "+str(r2_score(y_test, y_pred_ADA)))
print("Ada Boost Regression MSE value: "+str(mean_squared_error(y_test, y_pred_ADA)))


# Random forest seems like the best regression model. As a result I am going to validate the results of the Random Forest model for comparison with classification techniques. 

# In[338]:


r2_scores_random_forest = []
MSE_scores_random_forest = []

kf = KFold(n_splits=len(df_columns_list))

count = 0

for train_index, test_index in kf.split(df_train):
    
    clear_output()
    print(count)
    count += 1 
    
    X_train, X_test = train[train_index], train[test_index]
    y_train, y_test = labels[train_index], labels[test_index]
    
    #Random Forest 
    RF = RandomForestRegressor(n_estimators=20, random_state=36)
    RF.fit(X_train, y_train.ravel())
    y_pred_RF = RF.predict(X_test)
    r2_scores_random_forest.append(r2_score(y_test, y_pred_RF))
    MSE_scores_random_forest.append(mean_squared_error(y_test, y_pred_RF))
        
print("Random Forest Regression validated R^2 value: "+str(np.mean(r2_scores_random_forest)))
print("Random Forest Regression MSE value: "+str(np.mean(MSE_scores_random_forest)))


# # Classification of Loan grade #

# I am now going to try and apply classifaction algorithms to predict loan grades. I am going to start with a support vector machine classifier. I validate the initial models here, as they can be trained faster, but with a smaller number of folds. 

# In[339]:


labeller = preprocessing.LabelEncoder()
grade_targets = (np.array(grade_targets)).tolist()
labeller.fit(grade_targets)
grades_encoded = labeller.transform(grade_targets)


# In[340]:


count = 0 
SVC_accuracies = [] 

kf = KFold(n_splits=3)

for train_index, test_index in kf.split(df_train):
    
    clear_output()
    print(count)
    count += 1 
    
    X_train, X_test = train[train_index], train[test_index]
    y_train, y_test = grades_encoded[train_index], grades_encoded[test_index]
    
    #Support Vector Classification
    SVC_model = LinearSVC(C=1.0)
    SVC_model.fit(X_train, y_train)
    SVC_pred = SVC_model.predict(X_test)
    SVC_accuracies.append(accuracy_score(y_test, SVC_pred))
    
print("Accuracy of SVC "+str(np.mean(SVC_accuracies)))


# Poor results are achieved with SVC. Attempts will now be made with KNN algorithm. 

# In[343]:


KNN_accuracies = [] 

kf = KFold(n_splits=3)

for train_index, test_index in kf.split(df_train):
    
    clear_output()
    print(count)
    count += 1 
    
    X_train, X_test = train[train_index], train[test_index]
    y_train, y_test = grades_encoded[train_index], grades_encoded[test_index] 
    
    #K Nearest Neighbour Classification
    neigh = KNeighborsClassifier(n_neighbors=400)
    neigh.fit(X_train, y_train)
    KNN_pred = neigh.predict(X_test)
    KNN_accuracies.append(accuracy_score(y_test, KNN_pred))
    
print("Accuracy of KNN "+str(np.mean(KNN_accuracies)))


# Due to the poor results of the past two algorithms I am now going to try more powerful ensemble methods of classification. 

# In[344]:


count = 0 
RFC_accuracies = [] 

kf = KFold(n_splits=3)

for train_index, test_index in kf.split(df_train):
    
    clear_output()
    print(count)
    count += 1 
    
    X_train, X_test = train[train_index], train[test_index]
    y_train, y_test = grades_encoded[train_index], grades_encoded[test_index] 
    
    #K Nearest Neighbour Classification
    RFC = RandomForestClassifier(max_depth=5, random_state=0)
    RFC.fit(X_train, y_train)
    RFC_pred = RFC.predict(X_test)
    RFC_accuracies.append(accuracy_score(y_test, RFC_pred))
    
print("Accuracy of Random Forest Classifier "+str(np.mean(RFC_accuracies)))


# In[345]:


count = 0 
GBC_accuracies = [] 

kf = KFold(n_splits=3)

for train_index, test_index in kf.split(df_train):
    
    clear_output()
    print(count)
    count += 1 
    
    X_train, X_test = train[train_index], train[test_index]
    y_train, y_test = grades_encoded[train_index], grades_encoded[test_index] 
    
    #Gradient Boosted Classification
    GBC = GradientBoostingClassifier(max_depth=5, random_state=0)
    GBC.fit(X_train, y_train)
    RFC_pred = RFC.predict(X_test)
    RFC_accuracies.append(accuracy_score(y_test, RFC_pred))
    
print("Accuracy of Gradient Boosting Classifier "+str(np.mean(RFC_accuracies)))


# After achieveing poor results with a variety of classification models it seems clear that it is best to use a Random Forest Regression model to predict Interest rates and use this to find loan grade. The use of natural language processing will be explored to see if any significant results can be achieved with this.

# # Classification of Loan Grade from Purpose Text Field# 
# 
# I hypothesise that some interesting information could be extracted from the purpose statement of the loans. Due to the complexity of the text passages I think a neural network approach will be most likely to succeed. I am going to build a Convolutional Neural Network that aims to predict loan grades from purpose statements. If it successful I will combine it with the predictions from the numerical models. 
# 
# Convolutional neural networks have been shown to be useful in text classification. For example as shown here: https://arxiv.org/pdf/1408.5882.pdf. I have used code from a tutorial by Richard Liao found here: https://richliao.github.io/supervised/classification/2016/11/26/textclassifier-convolutional/ that implements a CNN for text classification using Keras with a Tensorflow backend. An pre-trained word vector was downloaded for encoding the words from the loan statements. The word vector was downloaded here: https://nlp.stanford.edu/projects/glove/.

# In[21]:


#getting rid of NaNs
df_orig = df_orig[pd.notnull(df_orig['grade'])]


# In[22]:


from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical

from keras.layers import Embedding
from keras.layers import Dense, Input, Flatten
from keras.layers import Conv1D, MaxPooling1D, Embedding, Merge, Dropout
from keras.models import Model


# In[23]:


#converting data to list format
grade_targets = df_orig['grade']
grade_targets = (np.array(grade_targets)).tolist()


# In[24]:


#encoding labels
labeller.fit(grade_targets)
grades_encoded = labeller.transform(grade_targets)


# In[25]:


raw_labels = grades_encoded
raw_texts = df_orig['desc'].tolist()


# In[26]:


#setting the max number of words, sequence length, validation split, f
MAX_NB_WORDS = 20000
MAX_SEQUENCE_LENGTH = 10000
VALIDATION_SPLIT = 0.25

#making sure all values in training and testing sets are represented by strings 
texts = [str(i) for i in raw_texts]
labels = [str(i) for i in raw_labels]


# In[27]:


#text preprocessing with Keras 
tokenizer = Tokenizer(nb_words=MAX_NB_WORDS)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)

word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

#padding sequences by how much smaller they are than the Max sequence
data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)


# In[28]:


#converting labels to categorical variables
labels = to_categorical(np.asarray(labels))
print('Shape of data tensor:', data.shape)
print('Shape of label tensor:', labels.shape)


# In[29]:


#setting indices and shuffling data set 
indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]
nb_validation_samples = int(VALIDATION_SPLIT * data.shape[0])


# In[30]:


#separating loan descriptions and grades into training and testing sets. 
x_train = data[:-nb_validation_samples]
y_train = labels[:-nb_validation_samples]
x_val = data[-nb_validation_samples:]
y_val = labels[-nb_validation_samples:]

print('Number of loans of different grades in training and validation data set ')
print(y_train.sum(axis=0))
print(y_val.sum(axis=0))


# In[31]:


#loading pre-trained word vector into memory.
GLOVE_DIR = "/home/augustine/Downloads/glove.6B.100d.txt"
embeddings_index = {}
f = open(GLOVE_DIR)
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

print('Total %s word vectors in Glove 6B 100d.' % len(embeddings_index))


# In[32]:


#using Keras tools to embed words from loan descriptions
embedding_matrix = np.random.random((len(word_index) + 1, 100))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector
        
embedding_layer = Embedding(len(word_index) + 1,
                            100,
                            weights=[embedding_matrix],
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=True)


# In[33]:


#embedding input data
sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
embedded_sequences = embedding_layer(sequence_input)
#construction of the layers of the neural network. 
l_cov1= Conv1D(128, 5, activation='relu')(embedded_sequences)
l_pool1 = MaxPooling1D(5)(l_cov1)
l_cov2 = Conv1D(128, 5, activation='relu')(l_pool1)
l_pool2 = MaxPooling1D(5)(l_cov2)
l_cov3 = Conv1D(128, 5, activation='relu')(l_pool2)
l_pool3 = MaxPooling1D(35)(l_cov3)  # global max pooling
l_flat = Flatten()(l_pool3)
l_dense = Dense(128, activation='relu')(l_flat)
preds = Dense(7, activation='softmax')(l_dense)


# In[34]:


#building model in memory
model = Model(sequence_input, preds)
model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['acc'])


# In[35]:


#fitting model and printing stats
print("model fitting - simplified convolutional neural network")
model.summary()
model.fit(x_train, y_train, validation_data=(x_val, y_val),
          nb_epoch=10, batch_size=128)


# The model does not seem to classify loan description by more than an accuracy of 30%. This could be a significant result as they are 7 categories to choose from. However, more than likely the model has just picked up on the fact that some grades are more likely than others due to the different frequencies of loan grades. 
# 
# In further study, the use of a character-level convolutional neural network could be used: https://arxiv.org/pdf/1509.01626.pdf. Such methods have proven to be very sucessful. However, even these more complex models would probably require more data and layers of the neural network. 

# # Conclusion # 
# A random forest regressor seems to be the best method for interest rate prediction with an $r^{2}$ value of 0.81 and a MSE of 2.61. If you can predict interest rate prediction then it seems you can predict loan grades as seen here: https://www.lendingclub.com/foliofn/rateDetail.action
# 
# In further study, hyperparameters of the random forest could be tuned. Furthermore, output from superior NLP models could be used to aid decisions in the classification of loan grades. 
# 
# In addition different approaches to feature engineering could be explored to improve the accuracies of the models used.
