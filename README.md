# Creating the Titanic project using Pandas, Numpy, Seaborn, Matplotlib


## In this github repository I will showcase the step by step guide I took to create a project in a jupyter file:

1- First step is to import the libraries such as Pandas, Numpy, Seaborn, Matplotlib in this case as well as read both of the csv files. 
  - I have also used the .head() method to run the first five rows for both of the csv files.

  ```
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

data_test = pd.read_csv('titanic/test.csv')
data_train = pd.read_csv('titanic/train.csv')

  ```
2- The second step was to rename the columns from the train csv file to make them more readable. 

  ```
data_train = data_train.rename(columns={'Name':'Full Name','Pclass':'Class'})
data_train.head()

  ```

3- The next step was to find out any missing values that exist in dataset and with that, create a heatmap. 


  ```
data_train.isnull().sum()

plt.figure(figsize=(10,6))
sns.heatmap(data_train.isnull().astype(int), 
            cbar=False, 
            cmap=sns.color_palette(['deepskyblue', 'palegreen'])) 
plt.title('Heatmap of missing values')
plt.show()


  ```
<img width="659" alt="Screenshot 2025-05-04 at 11 24 10" src="https://github.com/user-attachments/assets/a5f75ab4-339d-4c98-a0ea-ca654b24c236" />


4- After creating the heatmap, I went on to create the percentage pie chart to compare the difference in females and males on the ship:

  ```
plt.pie([male,female],labels=['Male','Female'],autopct='%1.1f%%')
plt.title('Total Male & Female Passengers')
plt.show()

  ```
<img width="313" alt="Screenshot 2025-05-04 at 11 26 41" src="https://github.com/user-attachments/assets/7b85bfaa-1e75-4b27-b73c-a9185176e5b1" />


5- The following step was to continue using the pie charts in order to create the survival percentage of men and women, as follows:

 ```
plt.pie([percentage2,100-percentage2],labels=['Survived','Not Survived'],autopct='%1.1f%%',colors= ['green','red'])
plt.title('Women Survived')
plt.show()

  ```

<img width="325" alt="Screenshot 2025-05-04 at 11 27 58" src="https://github.com/user-attachments/assets/8422de00-00f6-46f4-86d0-fabb899c534e" />

 ```
men_data = data_train.loc[data_train.Sex =='male']['Survived']

rate_men = sum(men_data)/len(men_data)


percentage1 = int(rate_men * 100)
print(percentage1)

plt.pie([percentage1,100-percentage1],labels=['Survived','Not Survived'],autopct='%1.1f%%',colors= ['green','red'])
plt.title('Men Survived')
plt.show()

  ```
<img width="357" alt="Screenshot 2025-05-04 at 11 28 33" src="https://github.com/user-attachments/assets/d56583d1-c57a-4360-a4df-06f1265dc008" />



6- For the presentation sake, the creation of a bar chart with the amounts of men and women was also created:


 ```
plt.figure(figsize=(6, 6))
plt.title('Women vs Men Survived')
plt.bar(['Women','Men'],[percentage2,percentage1], color =['red','blue'])
plt.ylabel('Percentage Survived')
plt.xlabel('Gender')
plt.grid(False)
plt.show()

  ```

<img width="423" alt="Screenshot 2025-05-04 at 11 29 44" src="https://github.com/user-attachments/assets/f35c1b50-f113-4ff7-a19d-7659d4939270" />



7 - After calculating the amount on the ship, we couldn't forget to create a pie chart for the amount of children that were on the ship, comparing to the adults:

 ```
adults = data_train.loc[data_train.Age >= 18]['Age'].count()
print(adults)

children = data_train.loc[data_train.Age <18]['Age'].count()
print(children)

plt.pie([adults,children],labels=['Adults','Children'],autopct='%1.1f%%',colors= ['Orange','Red'])
plt.title('Adults vs Children')
plt.show()

  ```

<img width="323" alt="Screenshot 2025-05-04 at 11 31 21" src="https://github.com/user-attachments/assets/dce03466-52a0-4b25-b36b-8ac86fb3f9cd" />

8 - For a more clear view, a bar graph has also been created to show the total amount of passengers with each one split into their own category:

 ```
total_amount_ppl = data_train.loc[data_train.Survived == 1]['Survived'].count()
print(total_amount_ppl)

survivors = [survived_women,survived_men,survived_children,total_amount_ppl]

data_series = pd.Series(survivors,index=['Women','Men','Children','Total Number'])

data_graph = data_series.plot(kind='bar',title='Amount of passanger survived',xlabel='Gender',ylabel='Amount', x='data',y='1000',color = ['violet','blue','gold','black'])

for i, value in enumerate(survivors):
    plt.text(i, value + 10, str(value), ha='center', fontsize=10)  # Adjust +10 if needed for spacing

plt.ylim(0, max(survivors) + 50)  # Optional: add space above bars
plt.show()


  ```

<img width="462" alt="Screenshot 2025-05-04 at 12 52 59" src="https://github.com/user-attachments/assets/7562cf1c-a191-4805-9923-762955eab08d" />


9 - The following step was to create a line graph showcasing the survival rate in percentages between the classes. The below were taken to achieve this:


 ```
survived_1st = data_train.loc[(data_train.Survived == 1) & (data_train.Class == 1),'Class'].count()

total_1st = data_train.loc[data_train.Class == 1]['Class'].count()

percentage_1st = int((survived_1st/total_1st *100))
print(f"{percentage_1st}% survived in first class")

  ```



 ```
survived_2nd = data_train.loc[(data_train.Survived == 1) & (data_train.Class == 2), 'Class'].count()

total_2nd = data_train.loc[data_train.Class == 2]['Class'].count()
percentage_2nd = int((survived_2nd/total_2nd *100))
print(f"{percentage_2nd}% survived in second class")

  ```

 ```
survived_3rd = data_train.loc[(data_train.Survived == 1)& (data_train.Class == 3), 'Class'].count()
total_3rd = data_train.loc[data_train.Class == 3]['Class'].count()

percentage_3rd = int((survived_3rd/total_3rd*100))
print(f"{percentage_3rd}% survived in third class")
  ```

 ```
x = [1, 2, 3]

y = [percentage_1st, percentage_2nd, percentage_3rd]

plt.figure(figsize=(10, 6))
plt.plot(x, y, marker='o')
plt.title('Percentage of People Survived in Each Class')
plt.xlabel('Class')
plt.ylabel('Percentage')
plt.xticks([1, 2, 3])
plt.ylim(0, 100)
plt.grid(True)
plt.show()
  ```

<img width="683" alt="Screenshot 2025-05-04 at 11 36 04" src="https://github.com/user-attachments/assets/0b124987-ad1e-4c49-b44b-2a2231c9e96d" />



10 - Like for my previous graphs, after creating percentage amounts, sum amount in digits graph has also been created:


 ```
classes = [1, 2, 3]
values = [survived_1st, survived_2nd, survived_3rd]
colors = ['gold', 'green', 'brown']

# Create the bar chart
plt.bar(classes, values, color=colors)
plt.title('Amount of People That Survived per Class')
plt.xlabel('Class')
plt.ylabel('Amount')
plt.xticks(classes)

# Add labels above bars
for i, val in enumerate(values):
    plt.text(classes[i], val + 5, str(val), ha='center', fontsize=10)  # Adjust +5 for spacing

plt.ylim(0, max(values) + 50)  # Optional: ensure space above bars
plt.show()
  ```

<img width="451" alt="Screenshot 2025-05-04 at 11 38 55" src="https://github.com/user-attachments/assets/92f8c86b-a323-4296-8f11-a737479e7755" />



11 - Moving on to a more technical approach, I wanted to use double bars in order to create a more varied and more understandable outcome:


 ```
class_category = ['First Class','Second Class','Third Class']

class_full = [first_class_full,second_class_full,third_class_full]

class_died = [first_class_died,second_class_died,third_class_died]

x = range(len(class_category))
bar_width = 0.3

plt.bar(x,class_full,width=bar_width,label='Total Amount',color='skyblue')
plt.bar([p + bar_width for p in x], class_died, width=bar_width, label='Non Survivals', color='green')
plt.title('Amount of people in each class')
plt.xlabel('Class')
plt.ylabel('Amount')
plt.xticks([p + bar_width/2 for p in x], class_category)
plt.legend()
plt.show() 
  ```

<img width="457" alt="Screenshot 2025-05-04 at 11 40 20" src="https://github.com/user-attachments/assets/5252ca9c-b464-4fef-8a4d-43fecf1dc6f8" />

12 - Creating a visual pie chart to show the percentage of people in each class also helped us to understand everything more clearly.



 ```
first_class = data_train.loc[data_train.Class == 1]['Class'].count()
second_class = data_train.loc[data_train.Class == 2]['Class'].count()
third_class = data_train.loc[data_train.Class == 3]['Class'].count()

first_percentage = int((first_class/total_amount_ppl *100))
second_percentage = int((second_class/total_amount_ppl *100 +1))
third_percentage = int((third_class/total_amount_ppl *100))

total_percentage = first_percentage,second_percentage,third_percentage

plt.pie(total_percentage,labels=['First Class','Second Class','Third Class'],autopct='%1.1f%%',colors= ['gold','green','brown'])
plt.title('Class Distribution')
plt.show()
  ```
<img width="310" alt="Screenshot 2025-05-04 at 11 45 57" src="https://github.com/user-attachments/assets/16d789ea-5d07-47a3-8bf9-42e16a642ca0" />


13 - The next step was to create a scatter plot comparing the age vs survival rate. This was my first time creating a scatter plot. 


 ```
plt.figure(figsize=(10, 8))
sns.scatterplot(x='Age', y='Fare', hue='Survived', data=data_train, palette='viridis', size='Class', sizes=(50, 200), alpha=0.7)
plt.title('Age vs Fare by Survival Status')
plt.xlabel('Age')
plt.ylabel('Fare')
plt.legend(title='Survived')
plt.savefig('age_fare_scatter.png')
plt.show()

  ```
<img width="677" alt="Screenshot 2025-05-04 at 11 49 51" src="https://github.com/user-attachments/assets/f22c5137-3352-4422-8d53-f91c218944e2" />

14 - Whilst creating the graph regarding the amount of people that embarked on each station has allowed me to learn how to create the labels, as follows:

 ```
embarks = [count_Cherbourg,count_Southampton,count_Queenstown]
data_series = pd.Series(embarks,index=['Cherbourg','Southampton','Queenstown'])

data_graph = data_series.plot(kind='bar',title='Amount of people that embarked on each station',xlabel='Harbour',ylabel='Amount', x='data',y='1000',color = ['orange','blue','lime'])


for i, value in enumerate(embarks):
    plt.text(i, value + 10, str(value), ha='center', fontsize=10)  # Adjust +10 if needed for spacing

plt.ylim(0, max(embarks) + 50)  # Optional: add space above bars
plt.show()

  ```
<img width="453" alt="Screenshot 2025-05-04 at 11 52 12" src="https://github.com/user-attachments/assets/16baa86c-d151-45e1-a428-03a40219b113" />



15 - Like previously created, two bar charts are made next to each other. Managing the bar width is very important in order for the bars not to overlap.

 ```
categories = ['Cherbourg','Southampton','Queenstown']
total_counts = [count_Cherbourg,count_Southampton,count_Queenstown]
embark_survived = [surived_Cherbourg,survived_Southampton,survived_Queenstown]

bar_width = 0.2 
x = range(len(categories))

plt.bar(x, total_counts, width=bar_width, label='Total', color='skyblue')
plt.bar([p + bar_width for p in x], embark_survived, width=bar_width, label='Survived', color='green')

plt.xlabel('Embark Harbour')
plt.ylabel('Count')
plt.title('Total vs Embark Harbour')
plt.xticks([p + bar_width/2 for p in x], categories)
plt.legend()

plt.tight_layout()
plt.show()

  ```
<img width="502" alt="Screenshot 2025-05-04 at 11 58 59" src="https://github.com/user-attachments/assets/1fba2269-4308-4378-8bfd-3c3df5422df1" />



16 - One of the most interesting information that I have learnt are the pallette colours use in seaborn and in matplotlib. Using them effectively clearly helps in making the visualisation more eye-appealing. 

 ```
values = [one_person_family,two_people_family,three_people_family,four_people_family,five_people_family]

siblings = ['0 siblings','1 sibling','2 siblings','3 siblings','4 siblings']

colors = ['mediumspringgreen','springgreen','limegreen','green','darkgreen']

plt.figure(figsize=(10, 6))
bars = plt.bar(siblings,values,color = colors)
plt.title('Amount of people with different number of siblings')
plt.xlabel('Amount')
plt.ylabel('Number of siblings')
plt.show()

  ```
<img width="678" alt="Screenshot 2025-05-04 at 12 00 45" src="https://github.com/user-attachments/assets/5e1d3787-c4ad-428b-831b-5be5f35eda50" />

17 - Below is another example of using double bars but using a different bar width due to the fact that there were more x axis inputs.

 ```
siblings = ['0 siblings','1 sibling','2 siblings','3 siblings','4 siblings']

values = [one_person_family,two_people_family,three_people_family,four_people_family,five_people_family]

died_values = [one_person_family_survived,two_people_family_survived,three_people_family_survived,four_people_family_survived,five_people_family_survived]

bar_width = 0.3 
x = range(len(siblings))

plt.bar(x,values,width=bar_width,label='Total Amount',color='skyblue')
plt.bar([p + bar_width for p in x], died_values, width=bar_width, label='Non Survivals', color='green')
plt.title('Amount of people with different number of siblings')
plt.xlabel('Number od siblings')
plt.ylabel('Amount')
plt.xticks([p + bar_width/2 for p in x], siblings)
plt.legend()
plt.show()


  ```
<img width="457" alt="Screenshot 2025-05-04 at 12 04 01" src="https://github.com/user-attachments/assets/8d56cab8-dbd9-4bd4-83bc-f84fac6afb74" />

18 - The following boxplot was the first one which was created by me in seaborn. Very effective way to create a clear way to present the average age of each gender. 


 ```
plt.figure(figsize=(10,6))
sns.boxplot(x='Sex', y='Age', data=data_train, palette = ['turquoise','hotpink'])
plt.xlabel('Gender')
plt.ylabel('Age')
plt.title('Age Distribution by Sex')


  ```
<img width="662" alt="Screenshot 2025-05-04 at 12 05 42" src="https://github.com/user-attachments/assets/63f6befc-97a9-42a7-a346-339e96f8a49a" />

19 - To practice even more, I have compared the class with age in order to compare the average age of passengers per class.


 ```
plt.figure(figsize=(10,6))
sns.boxplot(x='Class', y='Age', data=data_train,palette = ['bisque','goldenrod','darkorange'])
plt.xlabel('Class')
plt.ylabel('Age')
plt.title('Age Distribution by Class')


  ```

<img width="669" alt="Screenshot 2025-05-04 at 12 08 31" src="https://github.com/user-attachments/assets/2ad9634b-242e-4b9f-a35b-4bd6fab375fd" />


20 - One of the most expanded and my proudest graphs which I created in this project was the following:

 ```
age_groups = ['Below 12','13 - 17','18-30','31-49','50+']
age_survivors = [child,teen,young_adult,adults,senior]
original = [child_original,teen_original,young_adult_original,adults_original,senior_original]


bar_width = 0.3
x = range(len(age_groups))

plt.bar(x,original,width=bar_width,label = 'Original Amount',color = 'purple')
plt.bar([p + bar_width for p in x], age_survivors, width=bar_width, label='Survivors', color='green')
plt.title('Amount of people with different age groups')
plt.xlabel('Age Group')
plt.ylabel('Amount')
plt.xticks([p + bar_width/2 for p in x], age_groups)
plt.legend()
plt.show()

  ```

<img width="448" alt="Screenshot 2025-05-04 at 12 11 23" src="https://github.com/user-attachments/assets/d3ffd2c0-777a-4d02-93a9-3f3fb17c902a" />


21 - The final step was to use a countplot as a bar chart in order to find out the number of Parent and Children per passenger.

 ```
ax = sns.countplot(x='Parch', data = data_train, palette = ['lawngreen','darkgreen'])

ax.set_title('Parent/Child Amount Vs Passenger Count')


ax.set_xlabel('Parent/Child')
ax.set_ylabel('Passenger Count')


  ```
<img width="453" alt="Screenshot 2025-05-04 at 12 13 27" src="https://github.com/user-attachments/assets/bc6f4fda-fb93-4aa5-a616-8cda19eafbe8" />


## 22- The total project consisted of 20 graphs to analyse as much as possible. I am very proud to showcase this project to yourself. The skills implemented in this project has risen comparing to the previous Student analysis project. For any recommendations, please let me know! Thank you for reading.
