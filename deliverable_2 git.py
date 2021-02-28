#import libraries 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_samples, silhouette_score
import scikitplot as skplot
#load data
df = pd.read_csv("insurance.csv")
#check data structure and missing data
df.info()
df.describe()
df.head(3)
#EDA
#distribution of the charges 
df.charges.plot(kind="hist")
plt.xlabel('Amount in $')
plt.legend()
plt.show()
#distribution of people <= 35 and >35 (young adulthood age range 18-35, senior adult 36-55, elder>56)
fig = plt.figure(figsize=(20,10))
ax = fig.add_subplot(121)
ax = sns.distplot(df['charges'][df['age'] <= 35])
plt.title('Age <= 35')
plt.xticks([0,10000,20000,30000,40000,50000],np.arange(0,50000,10000))
plt.xlabel('Cost')

ax = fig.add_subplot(122)
ax = sns.distplot(df['charges'][df['age'] > 35])
plt.title('Age > 35')
plt.xticks([0,10000,20000,30000,40000,50000],np.arange(0,50000,10000))
plt.xlabel('Cost')

plt.show()
#plot charge by age
df['age_cat'] = np.nan
lst = [df]

for col in lst:
    col.loc[(col['age'] >= 18) & (col['age'] <= 35), 'age_cat'] = 'Young Adult'
    col.loc[(col['age'] > 35) & (col['age'] <= 55), 'age_cat'] = 'Senior Adult'
    col.loc[col['age'] > 55, 'age_cat'] = 'Elder'

# Means
avg_ya_charge = df["charges"].loc[df["age_cat"] == "Young Adult"].mean()
avg_sa_charge = df["charges"].loc[df["age_cat"] == "Senior Adult"].mean()
avg_e_charge = df["charges"].loc[df["age_cat"] == "Elder"].mean()

# Median
med_ya_charge = df["charges"].loc[df["age_cat"] == "Young Adult"].median()
med_sa_charge = df["charges"].loc[df["age_cat"] == "Senior Adult"].median()
med_e_charge = df["charges"].loc[df["age_cat"] == "Elder"].median()

fig = plt.figure(figsize=(20,10))
ax1 = fig.add_subplot(121)
ax1 = sns.barplot(x=['Young Adults', 'Senior Adults', 'Elder'], y=[avg_ya_charge, avg_sa_charge, avg_e_charge])
plt.title("Average Charge by Age")
plt.ylabel("Patient Charges")
plt.xlabel("Age Category")

ax2 = fig.add_subplot(122, sharey=ax1)
ax2 = sns.barplot(x=['Young Adults', 'Senior Adults', 'Elder'], y=[med_ya_charge, med_sa_charge, med_e_charge])
plt.title("Median Charge by Age")
plt.ylabel("Patient Charges")
plt.xlabel("Age Category")

plt.show()

#charges by region 
charges = df['charges'].groupby(df.region).sum().sort_values(ascending = True)
f, ax = plt.subplots(1, 1, figsize=(8, 6))
ax = sns.barplot(charges, charges.index, palette="Blues")
plt.title("Charges by Region")
plt.show()

#modelling 
df["weight_condition"] = np.nan
lst = [df]

for col in lst:
    col.loc[col["bmi"] < 18.5, "weight_condition"] = "Underweight"
    col.loc[(col["bmi"] >= 18.5) & (col["bmi"] < 24.986), "weight_condition"] = "Normal Weight"
    col.loc[(col["bmi"] >= 25) & (col["bmi"] < 29.926), "weight_condition"] = "Overweight"
    col.loc[col["bmi"] >= 30, "weight_condition"] = "Obese"
    

# Two subplots one with weight condition and the other with smoker.

f, (ax1, ax2) = plt.subplots(ncols=2, figsize=(18,8))
sns.scatterplot(x="bmi", y="charges", hue="weight_condition", data=df, palette="Set1", ax=ax1)
ax1.set_title("Relationship between Charges and BMI by Weight Condition")

sns.scatterplot(x="bmi", y="charges", hue="smoker", data=df, palette="Set1", ax=ax2)
ax2.set_title("Relationship between Charges and BMI by Smoking Condition")

plt.show()

#Kmeans model 
X = df[["bmi", "charges"]]
sc = StandardScaler()
xs = sc.fit_transform(X)
X = pd.DataFrame(xs, index=X.index, columns=X.columns)

X.describe().T

KS = range(2, 6)

# storage
inertia = []
silo = []

for k in KS:
  km = KMeans(k)
  km.fit(X)
  labs = km.predict(X)
  inertia.append(km.inertia_)
  silo.append(silhouette_score(X, labs))

plt.figure(figsize=(15,5))


plt.subplot(1, 2, 1)
plt.title("Inertia")
sns.lineplot(KS, inertia)

plt.subplot(1, 2, 2)
plt.title("Silohouette Score")
sns.lineplot(KS, silo)

plt.show()

k3 = KMeans(3)
k3_labs = k3.fit_predict(X)

# metrics
k3_silo = silhouette_score(X, k3_labs)
k3_ssamps = silhouette_samples(X, k3_labs)
np.unique(k3_labs)

skplot.metrics.plot_silhouette(X, k3_labs, title="KMeans - 3", figsize=(15,5))
plt.show()

fig = plt.figure(figsize=(12,8))

plt.scatter(X.values[:,0], X.values[:,1], c=k3.labels_, cmap="Set1_r", s=25)
plt.scatter(k3.cluster_centers_[:,0] ,k3.cluster_centers_[:,1], color='black', marker="x", s=250)
plt.title("Kmeans Clustering \n Finding Unknown Groups in the Population", fontsize=16)
plt.show()