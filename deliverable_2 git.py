#import libraries 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
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

