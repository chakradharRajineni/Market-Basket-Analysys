#!/usr/bin/env python
# coding: utf-8

# In[26]:


import numpy as np 
import pandas as pd 

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


aisle=pd.read_csv('aisles.csv')
depts=pd.read_csv('departments.csv')
orders=pd.read_csv('orders.csv')
products=pd.read_csv('products.csv')
op_train=pd.read_csv('order_products__train.csv')
op_prior=pd.read_csv('order_products__prior.csv')


# In[3]:


op_train.head()


# In[9]:


op_train.shape


# In[4]:


orders.head()


# In[5]:


products.head()


# In[6]:


depts.head()


# In[7]:


aisle.head()


# ### Merging all DAtasets to get all values in a single table

# In[18]:


merge=pd.merge(op_train, products, on=['product_id', 'product_id'])
merge=pd.merge(merge, orders, on=['order_id', 'order_id'])
merge=pd.merge(merge, aisle, on=['aisle_id', 'aisle_id'])
merge=pd.merge(merge, depts, on=['department_id', 'department_id'])
merge.head()


# In[51]:


merge1=merge[['add_to_cart_order','aisle_id', 'product_id', 'department_id', 'order_number', 'reordered']]
merge1.head()


# ### Creating a pivot table mapping each user_id with Department, to define customer charecterstic

# In[19]:


cust_dept=pd.crosstab(merge['user_id'],merge['department'])
cust_dept.head(15)


# In[20]:


cust_dept.shape


# In[34]:


from sklearn.decomposition import PCA
pca = PCA(n_components=2)
pca.fit(cust_dept)
pca_samples = pca.transform(cust_dept)


# In[35]:


plt.figure()
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('Number of Components')
plt.ylabel('Variance (%)') 
plt.title('Pulsar Dataset Explained Variance')
plt.show()
print(pca.explained_variance_ratio_.sum())


# Since to cluster, we need to plot scatter plot and for this to plot it, only two components are taken to use as x-coordinate and y-coordinate. The reason first two PC's chosen because they would explain the maximum variability

# In[36]:


ps = pd.DataFrame(pca_samples)
ps.head()


# #### Scatter plot of Principal Components

# In[37]:


from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import proj3d
tocluster = pd.DataFrame(ps[[1,0]])
print (tocluster.shape)
print (tocluster.head())

fig = plt.figure(figsize=(8,8))
plt.plot(tocluster[1], tocluster[0], 'o', markersize=2, color='blue', alpha=0.5, label='class1')

plt.xlabel('x_values')
plt.ylabel('y_values')
plt.legend()
plt.show()


# #### Deciding K Based on K-Means 

# In[121]:


Nc = range(1, 10)
kmeans = [KMeans(n_clusters=i) for i in Nc]
kmeans
score = [kmeans[i].fit(cust_dept).score(cust_dept) for i in range(len(kmeans))]
score
plt.plot(Nc,score)
plt.xlabel('Number of Clusters')
plt.ylabel('Score')
plt.title('Elbow Curve')
plt.show()


# ##### using SSE

# In[125]:


sse = {}
for k in range(1, 10):
    kmeans = KMeans(n_clusters=k, max_iter=1000).fit(cust_dept)
    cust_dept["clusters"] = kmeans.labels_
    #print(data["clusters"])
    sse[k] = kmeans.inertia_ # Inertia: Sum of distances of samples to their closest cluster center
plt.figure()
plt.plot(list(sse.keys()), list(sse.values()))
plt.xlabel("Number of cluster")
plt.ylabel("SSE")
plt.show()


# From Above both, we can see k=4 should be the number of clusters

# ### KMeans Clustering

# In[126]:


from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

clusterer = KMeans(n_clusters=4,random_state=42).fit(tocluster)
centers = clusterer.cluster_centers_
c_preds = clusterer.predict(tocluster)
print(centers)


# In[127]:


fig = plt.figure(figsize=(8,8))
colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']
colored = [colors[k] for k in c_preds]
print (colored[0:11])
plt.scatter(tocluster[1],tocluster[0],  color = colored)
for ci,c in enumerate(centers):
    plt.plot(c[0], c[1], 'o', markersize=8, color='red', alpha=0.9, label=''+str(ci))

plt.xlabel('x_values')
plt.ylabel('y_values')
plt.legend()
plt.show()


# In[128]:


clust_prod = cust_dept.copy()
clust_prod['cluster'] = c_preds

clust_prod.head(10)


# In[142]:


print (clust_prod.shape)
f,arr = plt.subplots(2,2,sharex=True,figsize=(15,15))

c1_count = len(clust_prod[clust_prod['cluster']==0])

c0 = clust_prod[clust_prod['cluster']==0].drop(['cluster', 'clusters'],axis=1).mean()
arr[0,0].bar(range(len(clust_prod.drop(['cluster','clusters'],axis=1).columns)),c0)
c1 = clust_prod[clust_prod['cluster']==1].drop(['cluster','clusters'],axis=1).mean()
arr[0,1].bar(range(len(clust_prod.drop(['cluster','clusters'],axis=1).columns)),c1)
c2 = clust_prod[clust_prod['cluster']==2].drop(['cluster','clusters'],axis=1).mean()
arr[1,0].bar(range(len(clust_prod.drop(['cluster','clusters'],axis=1).columns)),c2)
c3 = clust_prod[clust_prod['cluster']==3].drop(['cluster','clusters'],axis=1).mean()
arr[1,1].bar(range(len(clust_prod.drop(['cluster','clusters'],axis=1).columns)),c3)
plt.show()


# In[143]:


c0.sort_values(ascending=False)[0:10]


# In[144]:


c1.sort_values(ascending=False)[0:10]


# In[145]:


c2.sort_values(ascending=False)[0:10]


# In[146]:


c3.sort_values(ascending=False)[0:10]


# #### We can see that all the clusters the most sold products respective departments are appearing but the order is changing and therefore, recommendations to each cluster would be different. We can differentiate more using Aisles as seen in the other code file

# In[ ]:


# Import `KMeans` module
from sklearn.cluster import KMeans

# Initialize `KMeans` with 4 clusters
kmeans=KMeans(n_clusters=4, random_state=123)

# Fit the model on the pre-processed dataset
kmeans.fit(wholesale_scaled_df)

# Assign the generated labels to a new column
wholesale_kmeans4 = wholesale.assign(segment = kmeans.labels_)

# Import the non-negative matrix factorization module
from sklearn.decomposition import NMF

# Initialize NMF instance with 4 components
nmf = NMF(4)

# Fit the model on the wholesale sales data
nmf.fit(wholesale)

# Extract the components 
components = pd.DataFrame(data=nmf.components_, columns=wholesale.columns)

# Group by the segment label and calculate average column values
wholesale_kmeans3.columns
kmeans3_averages = wholesale_kmeans3.groupby(['segment']).mean().round(0)

# Print the average column values per each segment
print(kmeans3_averages)

# Create a heatmap on the average column values per each segment
sns.heatmap(kmeans3_averages.T, cmap='YlGnBu')

# Display the chart
plt.show()

# Create the W matrix
W = pd.DataFrame(data=nmf.transform(wholesale), columns=components.index)
W.index = wholesale.index

# Assign the column name where the corresponding value is the largest
wholesale_nmf3 = wholesale.assign(segment = W.idxmax(axis=1))

# Calculate the average column values per each segment
nmf3_averages = wholesale_nmf3.groupby('segment').mean().round(0)

# Plot the average values as heatmap
sns.heatmap(nmf3_averages.T, cmap='YlGnBu')

# Display the chart
plt.show()

