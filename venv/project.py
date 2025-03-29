import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns

# _____________Most Runs Scored in IPL______________

# Load the CSV file
df = pd.read_csv("/Users/sujalkarki/Desktop/python learning/deliveries.csv")
print(df.head(10))      # Display the first 10 rows
print(df.columns)       # Display column names
print(df.info())       # Display dataframe info
print(df.shape)      # Display shape of dataframe

# Grouping by batsman to get total runs
batsman = df.groupby("batsman")["batsman_runs"].sum()

# Sorting in descending order and getting the top 15 scorers
runs = batsman.sort_values(ascending=False).head(15)

#plotting the bar graph for most runs scored 
plt.figure(figsize=(15,6))
sns.barplot(x=runs.index,y=runs,color="Red")
plt.xticks(rotation=90)
plt.title("Most Runs scored in the history of IPL",fontweight="bold")
plt.xlabel("Players name",fontweight="bold")
plt.ylabel("Runs Scored",fontweight="bold")

plt.show()


 # _____________Most 4's Hit in IPL______________

#checking how many runs scored by the batsman using 6
boundry_4=df.groupby("batsman")["batsman_runs"].agg(lambda x: ((x==4) .sum()))

#sorting in descending order and extracting the first 15 players to hit most 4's
run = boundry_4.sort_values(ascending=False).head(15)

#plotting the bar graph for most 4's hit 
plt.figure(figsize=(15,6))
sns.barplot(x=run.index,y=run,color="royalblue")
plt.xticks(rotation=90)
plt.title("Player who hit the most no. of 4's",fontweight="bold")
plt.xlabel("Players name",fontweight="bold")
plt.ylabel("No. of 4's",fontweight="bold")

plt.show()


# # _____________Most 6's Hit in IPL______________

#checking how many runs scored by the batsman using 6
boundry_6=df.groupby("batsman")["batsman_runs"].agg(lambda x :((x==4) .sum()))

#sorting in descending order and extracting the first 15 players to hit most 4's
run6=boundry_6.sort_values(ascending=False).head(15)


#plotting the bar graph for most 6's hit

plt.figure(figsize=(15,6))
sns.barplot(x=run6.index,y=run6,color="violet")
plt.xticks(rotation=90)
plt.title("Player who hit the most no. of 6's",fontweight="bold")
plt.xlabel("Players name",fontweight="bold")
plt.ylabel("No. of 6's",fontweight="bold")

plt.show()


#  ________Who has faced the most no. of dot balls in the IPL history so far______

#checking how many runs scored by the batsman using 0
dotball =df.groupby("batsman")["batsman_runs"].agg(lambda x:((x==0).sum()))

#sorting in descending order and extracting the first 15 players who faced most dot balls
dot =dotball.sort_values(ascending=False).head(15)
print(dot)

#plotting the bar graph for most dot balls faced 
plt.figure(figsize=(15,6))
sns.barplot(x=dot.index,y=dot.values,color="yellow")
plt.xticks(rotation=90)
plt.title("Player who faced the most no. of dot balls",fontweight="bold")
plt.xlabel("Players name",fontweight="bold")
plt.ylabel("No. of dot balls",fontweight="bold")

plt.show()



