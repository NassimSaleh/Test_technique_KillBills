#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
from plotnine import *
import plotnine
import psycopg2

# Import module for k-protoype cluster
from kmodes.kprototypes import KPrototypes

# Ignore warnings
import warnings
warnings.filterwarnings('ignore', category = FutureWarning)

# Format scientific notation from Pandas
pd.set_option('display.float_format', lambda x: '%.3f' % x)


# In[3]:


# Établir la connexion à la base de données PostgreSQL
conn = psycopg2.connect(
    host="prod-rds-db.cbijryjiwdgw.eu-west-3.rds.amazonaws.com",
    database="prodkbdb",
    user="testdata",
    password="testData678341A",
    port = 5432
)

# Créer un curseur pour exécuter les requêtes
cur = conn.cursor()

# Exécuter une requête SELECT pour récupérer les données de la table
cur.execute("SELECT * from items")
rows = cur.fetchall()

# Transformer les résultats en un tableau Python
result_array = []

for row in rows:    
    result_array.append(row)


# Fermer le curseur et la connexion à la base de données
cur.close()
conn.close()


# In[4]:


# transformer le tableau en array
result=np.array(result_array)


# transformer en dataframe et renommer les colonnes
data = pd.DataFrame(result, columns =['id', 'amount', 'description','date','itemName','parent','quantity','taxAmount','taxDescription','type','storeId','createdAt','updatedAt','taxRate'])

#supprimer les colonnes inutiles pour le clustering
data.drop(['id', 'date', 'parent', 'quantity','description', 'type','updatedAt'], axis = 1, inplace = True)

#changer le type de certaines colonnes
data['amount'] = data['amount'].astype(float)
data['taxAmount'] = data['taxAmount'].astype(float)
data['taxRate'] = data['taxRate'].astype(float)
data['createdAt'] = data['createdAt'].dt.strftime('%Y-%m-%d %H:%M:%S')

# retransformer en array
dataMatrix = data.to_numpy()

# Calculer le nombre de lignes à conserver (10% du total)
nombre_lignes_a_garder = int(0.1 * len(dataMatrix))

# Sélectionner aléatoirement 10% des indices de lignes
indices_lignes_gardees = np.random.choice(len(dataMatrix), size=nombre_lignes_a_garder, replace=False)

# Créer un nouveau tableau avec les lignes sélectionnées
nouveau_tableau = dataMatrix[indices_lignes_gardees]


# In[6]:


# Obtenir les index des variables catégorielles
catColumnsPos = [data.columns.get_loc(col) for col in list(data.select_dtypes('object').columns)]

# implémenter le modèle avec 4 clusters
kproto = KPrototypes(n_jobs = -1, n_clusters = 4, init = 'Huang', random_state = 0)
kproto.fit(nouveau_tableau , categorical = catColumnsPos)


# In[7]:


# la fonction demandée

def clustering(id):
    
    item1 = result[result[:, 0] == id] 
    

    item = pd.DataFrame(item1, columns =['id', 'amount', 'description','date','itemName','parent','quantity','taxAmount','taxDescription','type','storeId','createdAt','updatedAt','taxRate'])
    item.drop(['id', 'date', 'parent', 'quantity','description', 'type','updatedAt'], axis = 1, inplace = True)
    item['amount'] = item['amount'].astype(float)
    item['taxAmount'] = item['taxAmount'].astype(float)
    item['taxRate'] = item['taxRate'].astype(float)
    item['createdAt'] = item['createdAt'].dt.strftime('%Y-%m-%d %H:%M:%S')
    itemMatrix = item.to_numpy()

    cluster = kproto.predict( np.array(itemMatrix), categorical= catColumnsPos)  
    print("Le nouveau point appartient au cluster:", cluster)

# test de la fonction
clustering('06c520a0-802f-4e39-a99d-6f24d32de1e9')

