import os
import random
import tqdm
import numpy as np
import pandas as pd
import shap
import tensorflow as tf
from tensorflow import keras
from keras.models import *
from keras.layers import *
from keras.callbacks import *
from keras import backend as K
from keras.optimizers import Adam
from keras.layers import Dense, Embedding, SimpleRNN
from keras.models import Sequential
from scipy import stats
from pprint import pprint
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import *
from sklearn.utils import resample
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, mean_squared_error, mean_absolute_error
from sklearn.pipeline import make_pipeline

from pylab import plt, mpl
# Note: matplotlib style setting may warn in some environments
plt.style.use('seaborn')
mpl.rcParams['savefig.dpi'] = 300
mpl.rcParams['font.family'] = 'serif'
os.environ['PYTHONHASHSEED'] = '0'

import os
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

%matplotlib inline

np.set_printoptions(precision=4, suppress=True)

tf.random.set_seed(100)



# Get the list of available GPUs
gpus = tf.config.experimental.list_physical_devices('GPU')

# Limit GPU memory growth
physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device, True)

# If there is at least one GPU available, set it to be used
if gpus:
    try:
        # Set memory growth for each GPU
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        # Set visible devices to be used
        tf.config.set_visible_devices(gpus[0], 'GPU')
        # Print the available and logical GPUs
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # In case of any errors, fall back to CPU
        print(e)
        device = '/device:CPU:0'
else:
    print("No GPUs available, using CPU instead")
    device = '/device:CPU:0'



# If running in Colab, uncomment the drive mount lines and set the correct path.
try:
    from google.colab import drive
    DRIVE_AVAILABLE = True
except Exception:
    DRIVE_AVAILABLE = False

if DRIVE_AVAILABLE:
    print("Google Colab environment detected. Mounting Drive...")
    drive.mount('/drive', force_remount=False)
else:
    print("Google Colab drive not available. Make sure your data path is correct locally.")

# Data Importing
# Update this path to where your CSV is stored. In Colab it's: /drive/My Drive/Colab Notebooks/Data/equity_2000_2020.csv
data_path = '/drive/My Drive/Colab Notebooks/Data/equity_2000_2020.csv' if DRIVE_AVAILABLE else 'equity_2000_2020.csv'

dd = pd.read_csv(data_path, parse_dates=['DATE'])
dd.sort_values('DATE', inplace=True)
dd.set_index('DATE', inplace=True)

# Remove useless columns if present
if 'Unnamed: 0' in dd.columns:
    dd.drop(columns=['Unnamed: 0'], inplace=True)

# Filter data from 2016-07-29 to 2020-01-31
df = dd['2016-07-29':'2020-01-31'].copy()

df.head(10)


print(df.info())


# Total number of days in the dataset
Time_diff = df.index[-1] - df.index[0]
print(Time_diff.days, 'days in the dataset')
print(len(df['permno'].unique()), 'stocks in total in the dataset')

# Ratio of data for test
test_ratio = 0.25

# Index splitting train and test data
test_index = df.index[0] + (Time_diff*(1-test_ratio))

# Creating a new column of test flag
df['test_flag'] = False                                  # all false for training
df.loc[test_index : df.index[-1], 'test_flag' ] = True  # set test flag for period

# Visual Representation helper
def plot_stock(stockID):
    plt.figure(figsize=(9,6))
    df[(df['permno']==stockID)&(df['test_flag']==False)]['RET'].plot()
    df[(df['permno']==stockID)&(df['test_flag']==True)]['RET'].plot()
    plt.title('Example of Training/Testing Split')
    plt.legend(['Training','Testing'])
    plt.show()

# Example plot (uncomment to run)
# plot_stock(85314)



sequence_length = 8

def set_seed(seed):
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    random.seed(seed)

# LSTM input features
def gen_sequence(id_df, seq_length, seq_cols):
    data_matrix = id_df[seq_cols].values
    num_elements = data_matrix.shape[0]
    for start, stop in zip(range(0, num_elements-seq_length), range(seq_length, num_elements)):
        yield data_matrix[start:stop, :]

# LSTM output features
def gen_labels(id_df, seq_length, label):
    data_matrix = id_df[label].values
    num_elements = data_matrix.shape[0]
    return data_matrix[seq_length:num_elements, :]

# Train/Test data
col = df.columns

X_train, X_test = [], []
y_train, y_test = [], []

for (stock,is_test), _df in df.groupby(['permno', 'test_flag']):
    for seq in gen_sequence(_df, sequence_length, col):        
        if is_test:
            X_test.append(seq)
        else:
            X_train.append(seq)
    for seq in gen_labels(_df, sequence_length, ['RET']):        
        if is_test:
            y_test.append(seq)
        else:
            y_train.append(seq)

X_train = np.asarray(X_train)
y_train = np.asarray(y_train)
X_test = np.asarray(X_test)
y_test = np.asarray(y_test)

print('shape of the input X and output y\ninput training data dimension: ')
print(X_train.shape)
print('\noutput training data dimension: ')
print(y_train.shape)
print('\ninput testing data dimension: ')
print(X_test.shape)
print('\noutput testing data dimension: ')
print(y_test.shape)

for i in range(min(3, X_train.shape[0])):
    print('input ', i, ': \n',X_train[i,:,0:3])
    print('==> output ', i, ': \n', y_train[i], '\n')

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train.reshape(-1,X_train.shape[-1])).reshape(X_train.shape)
X_test = scaler.transform(X_test.reshape(-1,X_test.shape[-1])).reshape(X_test.shape)

print("After scaling X_train shape:", X_train.shape)



# Construct model
set_seed(50)

inputs_ae = Input(shape=(X_train.shape[1:]))

encoded_ae = LSTM(256, activation='relu', return_sequences=True)(inputs_ae)
encoded_ae = Dropout(0.2)(encoded_ae)
encoded_ae = BatchNormalization()(encoded_ae)

decoded_ae = LSTM(128, activation='relu', return_sequences=True)(encoded_ae)
decoded_ae = Dropout(0.2)(decoded_ae)
decoded_ae = BatchNormalization()(decoded_ae)

out_ae = TimeDistributed(Dense(1))(decoded_ae)

sequence_autoencoder = Model(inputs_ae, out_ae)
sequence_autoencoder.compile(optimizer=Adam(learning_rate=0.001), loss='mse')

es = EarlyStopping(patience=15, verbose=2, min_delta=0.001, 
                       monitor='val_loss', mode='auto', restore_best_weights=True)

sequence_autoencoder.fit(X_train, X_train, validation_data=(X_train, X_train),
                             batch_size=256, epochs=200, verbose=1, callbacks=[es])



encoder = Model(inputs_ae, encoded_ae)
encoded_feature_train = encoder.predict(X_train)
encoded_feature_test = encoder.predict(X_test)

print('encoded features: ', encoded_feature_train.shape)

X_train_ = np.concatenate([X_train, encoder.predict(X_train)], axis=-1)
X_test_ = np.concatenate([X_test, encoder.predict(X_test)], axis=-1)

X_train_.shape, X_test_.shape


%%time
set_seed(50)

inputs = Input(shape=(X_train_.shape[1:]))
lstm = LSTM(256, return_sequences=True, dropout=0.5)(inputs, training=True)
lstm = LSTM(128, return_sequences=False, dropout=0.5)(lstm, training=True)
dense = Dense(50)(lstm)
out = Dense(1)(dense)

model = Model(inputs, out)
model.compile(loss='mse', optimizer='adam')

es = EarlyStopping(patience=10, verbose=2, min_delta=0.001, 
                   monitor='val_loss', mode='auto', restore_best_weights=True)
model.fit(X_train_, y_train, validation_data=(X_train_, y_train), 
          epochs=200, batch_size=256, verbose=1, callbacks=[es])



# COMPUTE STOCHASTIC DROPOUT 
scores = []
for i in tqdm.tqdm(range(0,100)):
    scores.append(mean_absolute_error(y_test, model.predict(X_test_).ravel()))

print(np.mean(scores), np.std(scores))

results = {'LSTM':None, 'Autoencoder+LSTM':None}
results['Autoencoder+LSTM'] = {'mean':np.mean(scores), 'std':np.std(scores)}
print(results)



# Evaluate model on test set
mse = model.evaluate(X_test_, y_test, verbose=0)
y_pred = model.predict(X_test_)
mae = mean_absolute_error(y_test, y_pred)

print(f"Autoencoder + LSTM -  MSE: {mse:.4f}")
print(f"Autoencoder + LSTM -  MAE: {mae:.4f}")



set_seed(50)

inputs = Input(shape=(X_train.shape[1:]))
lstm = LSTM(256, return_sequences=True, dropout=0.5)(inputs, training=True)
lstm = LSTM(128, return_sequences=False, dropout=0.5)(lstm, training=True)
dense = Dense(50)(lstm)
out = Dense(1)(dense)

model = Model(inputs, out)
model.compile(loss='mse', optimizer='adam')

es = EarlyStopping(patience=15, verbose=2, min_delta=0.001, monitor='val_loss', mode='auto', restore_best_weights=True)
model.fit(X_train, y_train, validation_data=(X_train, y_train), 
          epochs=200, batch_size=256, verbose=1, callbacks=[es])

# Compute stochastic dropout
scores = []
for i in tqdm.tqdm(range(0,100)):
    scores.append(mean_absolute_error(y_test, model.predict(X_test).ravel()))

print(np.mean(scores), np.std(scores))

results['LSTM'] = {'mean':np.mean(scores), 'std':np.std(scores)}

for key, value in results.items():
    print(key, ': ', value, '\n')

# Evaluate model on test set
mse = model.evaluate(X_test, y_test, verbose=0)
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)

print(f"Simple LSTM -  MSE: {mse:.4f}")
print(f"Simple LSTM -  MAE: {mae:.4f}")



with tf.device('/GPU:0'):

  X_train_xgb, X_test_xgb, y_train_xgb, y_test_xgb = train_test_split(
      df.iloc[:, 3:], 
      df['RET'], 
      test_size=0.2, shuffle=False)

  grid_search_params = {
      'colsample_bytree': [0.3, 0.7],
      'learning_rate': [0.01, 0.1, 0.2, 0.5],
      'n_estimators': [100],
      'subsample': [0.2, 0.5, 0.8],
      'max_depth': [2, 3, 5]
  }

  xg_grid_reg = xgb.XGBRegressor(objective= "reg:squarederror", tree_method='gpu_hist', gpu_id=0)

  grid = GridSearchCV(estimator=xg_grid_reg, param_grid=grid_search_params, scoring='neg_mean_squared_error', cv=4, verbose=1)

  grid.fit(X_train_xgb, y_train_xgb)

  print('\n\n\n#############      Result      #################')
  print("GridSearchCV")
  print("Best parameters found: ", grid.best_params_)
  print("\nLowest MSE found: ", -grid.best_score_,'\nand')
  print("Lowest MAE found: ", mean_absolute_error(y_test_xgb, grid.predict(X_test_xgb)), '\n')

  xg_reg = xgb.XGBRegressor(objective= "reg:squarederror", 
                            **grid.best_params_, tree_method='gpu_hist', gpu_id=0)

  xg_reg.fit(X_train_xgb, y_train_xgb)

  y_pred = xg_reg.predict(X_test_xgb)

  mse = mean_squared_error(y_test_xgb, y_pred)
  mae = mean_absolute_error(y_test_xgb, y_pred)
  print("MSE: %f" % (mse))
  print("MAE: %f" % (mae))



# Split features and target
X = df.drop(['RET', 'permno'], axis=1)
y = df[['RET', 'permno']]

# Split into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

# Construct a NN model by TensorFlow
with tf.device('/GPU:0'):
    model = Sequential()
    model.add(Dense(64, activation='relu', input_dim=X_train.shape[1]))
    model.add(Dense(2))
    model.compile(loss='mse', optimizer='adam')
    early_stop = EarlyStopping(monitor='val_loss', patience=10, verbose=1)
    model.fit(X_train, y_train, epochs=100, validation_data=(X_test, y_test), verbose=True, callbacks=[early_stop])

# Define the explainer
explainer = shap.KernelExplainer(model, X_train.iloc[:50,:])

# Calculate the SHAP values for each feature using 300 rows
shap_values = explainer.shap_values(X_train.iloc[:300,:])

# Visualize feature importance (uncomment to plot)
# shap.summary_plot(shap_values, X_train, plot_type="bar")



# === PUSH NOTEBOOK TO GITHUB FROM COLAB ===
# Replace these two variables, then run the cell and paste your PAT when requested.
GITHUB_USERNAME = "yourusername"
REPO = "your-repo"   # repo must already exist or be writable
BRANCH = "main"
FILENAME = "Asset_Pricing_LSTM_Autoencoder.ipynb"

import getpass, os, shutil
token = getpass.getpass('GitHub Personal Access Token (with repo scope): ')
os.environ['GITHUB_TOKEN'] = token

# Configure git
!git config --global user.email "you@example.com"
!git config --global user.name "Your Name"

# Clone the repo
!rm -rf {REPO}
!git clone https://$GITHUB_TOKEN@github.com/{GITHUB_USERNAME}/{REPO}.git

# Copy the notebook into the repo
shutil.copy("/mnt/data/" + FILENAME, REPO + "/" + FILENAME)

# Commit & push
%cd {REPO}
!git add "{FILENAME}"
!git commit -m "Add Asset Pricing notebook"
!git push https://$GITHUB_TOKEN@github.com/{GITHUB_USERNAME}/{REPO}.git {BRANCH}

# Cleanup
del os.environ['GITHUB_TOKEN']
print("Done. Check your GitHub repo.")

