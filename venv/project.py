import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns

# # _____________Most Runs Scored in IPL______________

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

# __________________Player with highest strike rate_________________
# Load the dataset
df = pd.read_csv("/Users/sujalkarki/Desktop/python learning/deliveries.csv")

# Grouping by batsman and summing their total runs
batsman = df.groupby("batsman")["batsman_runs"].sum().reset_index()

# Renaming for clarity
batsman.rename(columns={"batsman": "batsman_name"}, inplace=True)

# Counting total balls faced by each batsman
total_balls = df.groupby("batsman")["ball"].count().reset_index()
total_balls.rename(columns={"batsman": "batsman_name", "ball": "total_balls_faced"}, inplace=True)

# Performing an outer merge
test = batsman.merge(total_balls, how="outer", on="batsman_name")

#creating a new column strike rate and storing the strike rate of the players 
test["strike_rate"]=(test["batsman_runs"]/test["total_balls_faced"])*100

# Display final dataset
print(test.describe())

#rounding the strike rate column upto 2 decimal places
test["strike_rate"]=test["strike_rate"].round(2)

#sorting the strike rate column and extracting top 20 strike rate
test.sort_values(by=["strike_rate"],ascending=False).head(20)

#sorting the strike rate column by a condition that the batter must have faced more than 200 delivery 
test1=test[test["total_balls_faced"]>200].sort_values(by=["strike_rate"],ascending=False).head(20).reset_index(drop=True)

#plotting the strike rate after applying the condition
plt.figure(figsize=(15,6))
sns.barplot(data=test1,x="strike_rate",y="batsman_name",palette="autumn")
plt.xticks(rotation=90)
plt.title("Player with highest strike rate after facing more than 200 balls",fontweight="bold")
plt.xlabel("Strike Rate",fontweight="bold")
plt.ylabel("Players name",fontweight="bold")

plt.show()


# ............................Bowling Analysis .................................

# # ________________Most Balls Through Player in  IPL___________________
balls= df.groupby("bowler")["ball"].count()
balls_through=balls.sort_values(ascending=False).head(15)

#plotting the bowlers with the most no. of balls
plt.figure(figsize=(15,6))
sns.barplot(x=balls_through.index,y=balls_through,color="lightgreen")
plt.xticks(rotation=90)
plt.title("Player who Through the most no. of balls",fontweight="bold")
plt.xlabel("Players name",fontweight="bold")
plt.ylabel("No. of balls",fontweight="bold")

# ____________Most wicket takers in IPL_____________
# Filtering out dismissals that count as wickets (excluding 'run out' and 'retired hurt')
valid_dismissals = ["bowler", "caught", "lbw", "stumped", "caught and bowled", "hit wicket"]

# Counting wickets for each bowler
wickets = df[df["dismissal_kind"].isin(valid_dismissals)].groupby("bowler")["dismissal_kind"].count().reset_index()

# Renaming columns
wickets.rename(columns={"dismissal_kind": "total_wickets"}, inplace=True)

# Sorting by most wickets
wickets = wickets.sort_values(by="total_wickets", ascending=False).reset_index(drop=True)

# Display the top 20 wicket-takers
print(wickets.head(20))

# Plot the top 20 wicket-takers
plt.figure(figsize=(15,6))
sns.barplot(data=wickets.head(20), x="total_wickets", y="bowler", palette="viridis")
plt.xlabel("Total Wickets", fontweight="bold")
plt.ylabel("Bowler", fontweight="bold")
plt.title("Top 20 Wicket-Takers in IPL History", fontweight="bold")
plt.show()
