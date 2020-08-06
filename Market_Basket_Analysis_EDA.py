#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import  Image
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
sns.set(style='white')
color = sns.color_palette()
import os

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go


# In[2]:


aisle=pd.read_csv('aisles.csv')
depts=pd.read_csv('departments.csv')
orders=pd.read_csv('orders.csv')
products=pd.read_csv('products.csv')
op_train=pd.read_csv('order_products__train.csv')
op_prior=pd.read_csv('order_products__prior.csv')


# ### Lets Explore all the Data sets

# ### Aisle

# In[3]:


aisle.head()


# In[4]:


aisle.shape


# There are 134 rows in the table with each aisle reprasented by its aisle id

# ### Departments

# In[5]:


depts.head()


# In[6]:


depts.shape


# There are 21 departments in total with each one reprasented by its id.

# ### Products

# In[7]:


products.head()


# This dataset provides information on each product with unique product id, department it belongs and which aisle id it is currently placed

# In[8]:


products.shape


# ### Products by Department

# There are 49688 products available in instacart from 21 departments which are arranged in 132 aisles

# In[9]:


x=products.department_id.value_counts().to_frame().sort_values(by='department_id', ascending=False)
x=x.reset_index()
x.columns=['department_id','count']
x=x.merge(depts, how='inner', on='department_id')
x
#to_frame().


# In[10]:


plt.figure(figsize=(10,10))
prod_dept = pd.merge(products, depts, how='left', on='department_id')
temp_series=prod_dept['department'].value_counts()
labels = (np.array(temp_series.index))
sizes = (np.array((temp_series / temp_series.sum())*100))
plt.pie(sizes, labels=labels, 
        autopct='%1.1f%%', startangle=200)
plt.title("Departments distribution", fontsize=15)
plt.show()


# Personal care department has most number of products meat seafood has least number of products(considering other and bulk are some not defined products)

# ### Products by Aisle

# In[11]:


y=products.aisle_id.value_counts().to_frame().sort_values(by='aisle_id', ascending=False)
y=y.reset_index()
y.columns=['aisle_id','count']
y=y.merge(aisle, how='inner', on='aisle_id')
y.head(10)


# In[12]:


plt.figure(figsize=(10,10))
prod_aisle = pd.merge(products, aisle, how='left', on='aisle_id')
temp_series=prod_aisle['aisle'].value_counts().sort_values()
temp_series2=temp_series.head(10)
labels = (np.array(temp_series2.index))
sizes = (np.array((temp_series2 / temp_series2.sum())*100))
plt.pie(sizes, labels=labels, 
        autopct='%1.1f%%', startangle=200)
plt.title("Aisle distribution Top10", fontsize=15)
plt.show()


# ### Orders

# In[13]:


orders.head()


# The dataset provodes information on each order, with order_id and divided dataset based on prior set, train set and test set. We will be predicting reordered items only for Test set<br>
# <br>
# - order_dow reprasents day of the week
# - order_hour_of_day reprasents hour of the order
# - days_since_prior_order: all is said in the name

# In[14]:


users_unique=orders.user_id.unique()
len(users_unique)


# ##### The data has a total of 206209 users data

# In[15]:


#eval_set tell us about the order which is train, test or prior
order_set=orders.eval_set.value_counts()
sns.barplot(order_set.index, order_set.values, alpha=0.8, color=color[1])
plt.xlabel('type of order')
plt.ylabel('Count of set')
plt.title('Histogram of Eval Set')


# #### So, we have data of 3.214 million orders which are given as prior(meaning they are orders done before) <br>  we have a trainig set to build the model with 131k orders <br> we have a test set of 75000, where our model will be tested.

# In[16]:


print(orders.days_since_prior_order.mode())
print(orders.days_since_prior_order.median())
orders.boxplot('days_since_prior_order')


# Median of Days since prior order is 7. Therefore, people are ordering weekly

# In[17]:


plt.figure(figsize=(12,8))
sns.countplot(x="days_since_prior_order", data=orders, color=color[3])
plt.ylabel('Count', fontsize=12)
plt.xlabel('Days since prior order', fontsize=12)
plt.xticks(rotation='vertical')
plt.title("Frequency distribution by days since prior order", fontsize=15)
plt.show()


# In[18]:


plt.figure(figsize=(12,8))
sns.countplot(x="order_dow", data=orders, color=color[3])
plt.ylabel('Count', fontsize=12)
plt.xlabel('Days since prior order', fontsize=12)
plt.xticks(rotation='vertical')
plt.title("Frequency distribution by days since prior order", fontsize=15)
plt.show()


# In[19]:


plt.figure(figsize=(12,8))
sns.countplot(x="order_hour_of_day", data=orders, color=color[3])
plt.ylabel('Count', fontsize=12)
plt.xlabel('Days since prior order', fontsize=12)
plt.xticks(rotation='vertical')
plt.title("Frequency distribution by days since prior order", fontsize=15)
plt.show()


# In[20]:


order_matrix=orders.groupby(['order_dow','order_hour_of_day'])['order_number'].aggregate('count').reset_index()
order_matrix=order_matrix.pivot('order_dow','order_hour_of_day','order_number')
plt.figure(figsize=(15,8))
sns.heatmap(order_matrix)
plt.title('Heatmap by week with Hour vs weekday')
plt.show()


# From the heatmap, we can see that Saturday and sunday are the days where most number of ordes came and between 9Am and 5Pm the traffic is higher 

# #### Number of Orders by Customer

# we have a total of 206209 customers (with number of orders between 4 and 100)in the dataset who gave 3.21 million orders

# In[21]:


#So a total of 206209 customers are recorded,whose prior orders are divided into test and train set 
order_count=orders.groupby('user_id')['order_number'].aggregate(np.max).reset_index()
order_count=order_count.order_number.value_counts()
plt.figure(figsize=(30,10))
sns.barplot(order_count.index, order_count.values, alpha=0.8, color=color[1])
plt.xlabel('Number of Prior orders')
plt.ylabel('Count of customers')
plt.title('Histogram of Number of prior orders by customer')


# We can see that, the number of reorders by customers are highly concentrated in 4-25 range

# ### Order products prior

# In[22]:


op_prior.head()


# In[23]:


op_prior.shape


# In[24]:


len(op_prior.order_id.unique())


# The dataset is large as we could see that 32 million records are provided with 3.2 million orders, with each order split into each product in that order.<br> The dataset is sorted byorder_id, and then by add to cart order. <br> The information regarding each product is given, weather it is reordered or not

# ### Order products Training set

# In[25]:


op_train.head()


# In[26]:


op_train.shape


# In[27]:


len(op_train.order_id.unique())


# Training set has 1.38 million records with same information as in orders_prior with information regarding 131k orders

# ## Products Data Exploration

# Lets combine all datasets

# In[28]:


ic=pd.merge(op_prior, products, how='left', on='product_id')
ic=pd.merge(ic,depts, how='left', on='department_id')
ic=pd.merge(ic, aisle, how='left', on='aisle_id')
ic.head()


# ### Most Ordered Products

# In[29]:


most_ordered = ic.groupby('product_id')['reordered'].aggregate({'Total_reorders':'count'}).reset_index()
most_ordered = pd.merge(most_ordered,products, how='left', on='product_id')
most_ordered = most_ordered.sort_values(by='Total_reorders',ascending=False).reset_index()
most_ordered[:10]


# Banana is the most ordered product, which should be because they are sold per banana instead of kilogramsn or dozens <Br>
# Top 10 products all belong to Fruits or produce , which are part of daily meals.
# <br>Aisle 24 is the busiest one, with 8 out of 10 from top selling products belong to that aisle. Therefore, we can say that, this aisle should appear in every suggestion or in top in the display.

# ### Departments with most selling products

# In[35]:


busy_department=ic.groupby('department_id')['reordered'].aggregate({'Total_Orders_by_department':'count'})
busy_department=busy_department.sort_values(by='Total_Orders_by_department',ascending=False).reset_index()
busy_department=pd.merge(busy_department, depts, how='left', on='department_id')
busy_department[:10]


# In[36]:


department_grouped=busy_department.groupby(['department']).sum()['Total_Orders_by_department'].sort_values(ascending=False)
f, ax = plt.subplots(figsize=(12, 8))
plt.xticks(rotation=90)
sns.barplot(department_grouped.index, department_grouped.values)
plt.ylabel('Total Number of Orders', fontsize=13)
plt.xlabel('Departments', fontsize=13)
plt.title('sales by Department', fontsize=20)
plt.show()


# Produce is the busiest department, with about 25% or orders belong to it only. 

# ### Aisles with most selling products

# In[37]:


busy_aisle=ic.groupby('aisle_id')['reordered'].aggregate({'Orders_by_aisle':'count'})
busy_aisle=busy_aisle.sort_values(by='Orders_by_aisle',ascending=False).reset_index()
busy_aisle=pd.merge(busy_aisle, aisle, how='left', on='aisle_id')
busy_aisle[:10]


# In[40]:


aisle_grouped=busy_aisle.groupby(['aisle']).sum()['Orders_by_aisle'].sort_values(ascending=False)
f, ax = plt.subplots(figsize=(30, 8))
plt.xticks(rotation=90)
sns.barplot(aisle_grouped.index, aisle_grouped.values)
plt.ylabel('Total Number of Orders', fontsize=13)
plt.xlabel('Aisles', fontsize=13)
plt.title('sales by Aisle', fontsize=20)
plt.show()


# ### Important Aisles by Department

# In[41]:


ic_grouped = ic.groupby(["department", "aisle"])["product_id"].aggregate({'Total_products': 'count'}).reset_index()
ic_grouped.sort_values(by='Total_products', ascending=False, inplace=True)
fig, axes = plt.subplots(7,3, figsize=(20,45), gridspec_kw =  dict(hspace=2))
for (aisle, group), ax in zip(ic_grouped.groupby(["department"]), axes.flatten()):
    g = sns.barplot(group.aisle, group.Total_products , ax=ax)
    ax.set(xlabel = "Aisle", ylabel=" Count")
    g.set_xticklabels(labels = group.aisle,rotation=90, fontsize=12)
    ax.set_title(aisle, fontsize=15)


# ## Orders data Exploration

# To explore orders, lets concat both prior orders and train orders Data, to look at wholesome picture

# In[42]:


df=pd.concat([op_prior, op_train], axis=0)
df.shape


# In[43]:


df.head()


# In[44]:


df.reset_index()


# In[45]:


df.isnull().sum()


# ### Number of products per order

# In[46]:


df_grouped=df.groupby('order_id')['add_to_cart_order'].aggregate('max').reset_index()
x=df_grouped.add_to_cart_order.value_counts().to_frame()
x['Number_of_products']=x.index
x.columns=['Number_of_orders', 'Number_of_products']
x.sort_values('Number_of_orders', ascending=False).reset_index()


# In[47]:


x=pd.DataFrame(x)


# In[48]:


x.dtypes


# In[49]:


sns.set_style('whitegrid')
f, ax = plt.subplots(figsize=(20, 7))
plt.xticks(rotation='vertical')
sns.barplot(x='Number_of_products', y='Number_of_orders', data=x, orient='v')
plt.xlabel('Number of items per order')
plt.ylabel('Count')
plt.title('Count of products per order', fontsize=15)


# We can see that the distributiuion is peaked at 6 products per order with Right skewness. <br>
# Therefore, while expecting orders, since it is a count per order, we can expect aroung 6 items per orders, with meadian somewhere between 8 and 10

# ## Reorders

# #### Ratio of Reorders

# In[50]:


Ratio_reorders=len(df[df['reordered']==1])/len(df)
print(Ratio_reorders)

Ratio_not_reordered=1-Ratio_reorders
print(Ratio_not_reordered)


# We can therefore here see that, customers are reordering 59% of the products.<br> 
# -  we can see the repurchase value of each products <br>
# - promote the products based on repurchase scenario

# #### Reordered products

# - reordered sum is total number of instances where product is reordered
# - reorder total is total number of times product is brought
# - reorder probability is of times reordered, how many times the product is reordered

# In[ ]:


df_grouped = df.groupby("product_id")["reordered"].aggregate({'reorder_sum': sum,'reorder_total': 'count'}).reset_index()
df_grouped['reorder_probability'] = df_grouped['reorder_sum'] / df_grouped['reorder_total']
df_grouped = pd.merge(df_grouped, products[['product_id', 'product_name']], how='left', on=['product_id'])
df_grouped = df_grouped[df_grouped.reorder_total > 100].sort_values(['reorder_probability'], ascending=False)
df_grouped.head().reset_index()


# So, If a customer orders Chocolate Love Bar, we can with 92% probability say that they will reorder the product againa dn therefore, any sort of discounts are not needed or since they will creorder to buy this product, we can mix the product with other relative product to promote sales

# In[ ]:


df_grouped.sort_values('reorder_sum', ascending=False).head(3).reset_index()


# #### Most reordered products are also most ordered products too

# In[ ]:


df_grouped.sort_values('reorder_total', ascending=False).head(3).reset_index()


# In[ ]:


grouped_df = ic.groupby(["department"])["reordered"].aggregate("mean").reset_index()

plt.figure(figsize=(12,8))
sns.pointplot(grouped_df['department'].values, grouped_df['reordered'].values, alpha=0.8, color=color[2])
plt.ylabel('Reorder ratio', fontsize=12)
plt.xlabel('Department', fontsize=12)
plt.title("Department wise reorder ratio", fontsize=15)
plt.xticks(rotation='vertical')
plt.show()


# ### Add to cart Order

# In[ ]:


df1=df[df['add_to_cart_order']==1]
cart_grouped=df1.groupby('product_id')['add_to_cart_order'].aggregate({'first_ordered_sum':sum}).reset_index()
cart_grouped=pd.merge(cart_grouped,products, how='left', on='product_id')
cart_grouped.head()


# In[ ]:


ic["add_to_cart_order_mod"] = ic["add_to_cart_order"].copy()
ic["add_to_cart_order_mod"].ix[ic["add_to_cart_order_mod"]>70] = 70
grouped_df = ic.groupby(["add_to_cart_order_mod"])["reordered"].aggregate("mean").reset_index()

plt.figure(figsize=(12,8))
sns.pointplot(grouped_df['add_to_cart_order_mod'].values, grouped_df['reordered'].values, alpha=0.8, color=color[2])
plt.ylabel('Reorder ratio', fontsize=12)
plt.xlabel('Add to cart order', fontsize=12)
plt.title("Add to cart order - Reorder ratio", fontsize=15)
plt.xticks(rotation='vertical')
plt.show()


# In[ ]:




