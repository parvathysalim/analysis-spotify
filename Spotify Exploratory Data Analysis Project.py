#!/usr/bin/env python
# coding: utf-8

# In[6]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[7]:


df_tracks = pd.read_csv('D:/spotifydatasets/tracks.csv')
df_tracks.head()


# In[8]:


#check null values
# Null values

pd.isnull(df_tracks).sum()


# In[30]:


# Rows and columns
df_tracks.info()


# In[47]:


# 10 least popular songs present
sorted_df = df_tracks.sort_values('popularity',ascending = True).head(10)
sorted_df


# In[9]:


# Descriptive Statistics
df_tracks.describe().transpose()


# In[11]:


# 10 most popular songs
most_popular = df_tracks.query('popularity > 90', inplace = False).sort_values('popularity',ascending = False)
most_popular[:10]


# In[12]:


# Release date
df_tracks.set_index("release_date",inplace = True)
df_tracks.index = pd.to_datetime(df_tracks.index)
df_tracks.head()


# In[13]:


df_tracks[["artists"]].iloc[18]


# In[14]:


# Check the column names in your DataFrame
print(df_tracks.columns)


# In[15]:


# Calculate duration in seconds and create a new column
# Convert duration in ms to second
df_tracks["duration"] = df_tracks["duration_ms"].apply(lambda x : round(x/1000))
# Drop the original "duration_in_ms" column
df_tracks.drop("duration_ms", inplace = True, axis =1)


# In[16]:


# Print the first few rows of the updated DataFrame
df_tracks.duration.head()


# In[17]:


corr_df = df_tracks.drop(["key","mode","explicit"],axis = 1).corr(method="pearson")
plt.figure(figsize=(14,6))
heatmap = sns.heatmap(corr_df,annot = True,fmt=".1g",vmin = -1,center =0,cmap = "inferno", linewidths = 1,linecolor = "Black")
heatmap.set_title("correlation heatmap between variable")
heatmap.set_xticklabels(heatmap.get_xticklabels(),rotation =90)


# In[18]:


# Create regression plots
sample_df = df_tracks.sample(int(0.004*len(df_tracks)))


# In[19]:


print(len(sample_df))


# In[56]:


plt.figure(figsize=(10,6))
sns.regplot(data = sample_df,y = "loudness",x = "energy",color ="c").set(title = "Loudness vs Energy Correlation")


# In[20]:


plt.figure(figsize=(10,6))
sns.regplot(data = sample_df,y = "popularity",x = "acousticness",color ="c").set(title = "Popularity vs Acousticness Correlation")


# In[21]:


# Create new column year
df_tracks['dates'] = df_tracks.index.get_level_values('release_date')
df_tracks.dates = pd.to_datetime(df_tracks.dates)
years = df_tracks.dates.dt.year


# In[ ]:


# Create distribution plot to visualize the total no. of songs in each year since 1922
#pip insatll --user seaborn == 0.11.0


# In[22]:


sns.displot(years,discrete = True, aspect = 2,height = 5,kind ="hist").set(title="Number of songs per year")



# In[23]:


# Create bar plot to view duration of songs over year

total_dr = df_tracks.duration
fig_dims = (18,7)
fig,ax = plt.subplots(figsize=fig_dims)
fig = sns.barplot(x=years, y=total_dr, ax=ax, errwidth=False).set(title = "Year vs Duration")
plt.xticks(rotation=90)


# In[24]:


# Create line plot to view duration of songs over year

total_dr = df_tracks.duration
sns.set_style(style="whitegrid")
fig_dims=(10,5)
fig,ax = plt.subplots(figsize = fig_dims)
fig = sns.lineplot(x=years, y=total_dr, ax=ax).set(title="Year vs Duration")
plt.xticks(rotation=60)


# In[27]:


df_genre = pd.read_csv("D:/spotifydatasets/spotifyFeatures.csv")


# In[28]:


df_genre.head()


# In[29]:


plt.title("Duration of the songs in Different Genres")
sns.color_palette("rocket",as_cmap = True)
sns.barplot(y='genre',x = 'duration_ms',data = df_genre)
plt.xlabel("Duration in milli seconds")
plt.ylabel("Genres")


# In[31]:


sns.set_style(style = "darkgrid")
plt.figure(figsize =(10,5))
famous = df_genre.sort_values("popularity",ascending = False).head(10)
sns.barplot(y='genre',x='popularity',data = famous).set(title = "Top 5 Genre by Popularity")


# In[ ]:




