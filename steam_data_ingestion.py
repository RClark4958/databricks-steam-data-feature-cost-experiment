# Databricks notebook source
import requests
import time
import json
from azure.storage.blob import BlobServiceClient

# COMMAND ----------

def get_app_list():
    url = 'https://api.steampowered.com/ISteamApps/GetAppList/v2/'
    response = requests.get(url)
    data = response.json()
    apps = data['applist']['apps']
    return apps

# COMMAND ----------

def get_app_details(app_id):
    url = f'https://store.steampowered.com/api/appdetails?appids={app_id}'
    response = requests.get(url)
    data = response.json()
    if str(app_id) in data and data[str(app_id)]['success']:
        return data[str(app_id)]['data']
    else:
        return None

# COMMAND ----------

apps = get_app_list()
sample_apps = apps[:200] # retrieve first 200 apps
app_details_list = []
for app in sample_apps:
    details = get_app_details(app['appid'])
    if details:
        app_details_list.append(details)
        time.sleep(0.2)

# COMMAND ----------

data_json = json.dumps(app_details_list)

# COMMAND ----------

storage_account_name = "rcsteamdata"
container_name = "rc-container"
sas_token = <token>   

mount_point = f"/mnt/{container_name}"
# dbutils.fs.mount(
#     source = f"wasbs://{container_name}@{storage_account_name}.blob.core.windows.net",
#     mount_point = mount_point,
#     extra_configs = { f"fs.azure.sas.{container_name}.{storage_account_name}.blob.core.windows.net": sas_token }
# )

# COMMAND ----------

dbutils.fs.put(f"/mnt/{container_name}/steam_app_details.json", data_json, overwrite=True)

# COMMAND ----------

import pandas as pd
df = pd.read_json(f"/dbfs/mnt/{container_name}/steam_app_details.json")

# COMMAND ----------

df = pd.json_normalize(df.to_dict(orient="records"))
print(df.columns)
print(df.head())

# COMMAND ----------

if 'price_overview.final' in df.columns:
        df = df.dropna(subset=['price_overview.final'])

df['price'] = df['price_overview.final'] / 100

if 'release_date.date' in df.columns:
    df['release_date'] = pd.to_datetime(df['release_date.date'], errors='coerce')

df_model = df[['name', 'release_date', 'price', 'genres']]

# COMMAND ----------

import matplotlib.pyplot as plt

plt.figure(figsize=(8, 5))
plt.hist(df_model['price'].dropna(), bins=20)
plt.xlabel('Price ($)')
plt.ylabel('Number of Games')
plt.title('Price Distribution of Steam Games')
plt.show()

# COMMAND ----------

X = df_model[['genres', 'release_date']]
y = df_model['price']

# COMMAND ----------

from sklearn.preprocessing import MultiLabelBinarizer
import numpy as np

# COMMAND ----------

X['genres'] = X['genres'].apply(lambda x: [g['description'] for g in x] if isinstance(x, list) else [])

mlb = MultiLabelBinarizer()
genre_dummies = pd.DataFrame(mlb.fit_transform(X['genres']), columns=mlb.classes_, index=X.index)
X = pd.concat([X.drop('genres', axis=1), genre_dummies], axis=1)

# COMMAND ----------

X['release_year'] = X['release_date'].dt.year.fillna(0).astype(int)
X['release_month'] = X['release_date'].dt.month.fillna(0).astype(int)
X = X.drop('release_date', axis=1)

# COMMAND ----------

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

import mlflow
import mlflow.sklearn

with mlflow.start_run():
    mlflow.log_param('model_type', 'RandomForestRegressor')
    mlflow.log_param('n_estimators', 100)
    mlflow.log_param('random_state', 42)