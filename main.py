# Python
import streamlit as st
import pandas as pd
from datetime import datetime
from scipy.stats import zscore
import matplotlib.pyplot as plt
import numpy as np
import pulp
import json
import plotly.graph_objects as go
import plotly.express as px
from plotly_calplot import calplot
import plotly.figure_factory as ff
from sklearn.metrics import mean_squared_error
import streamlit_authenticator as stauth
import yaml
from yaml.loader import SafeLoader
from st_files_connection import FilesConnection
import os

st.set_page_config(layout="wide")


# set google cloud key
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = '/Users/linkding/Downloads/Flux.json'

with open('config.yaml') as file:
    config = yaml.load(file, Loader=SafeLoader)

print(config['cookie']['name'])
authenticator = stauth.Authenticate(
    config['credentials'],
    config['cookie']['name'],
    config['cookie']['key'],
    config['cookie']['expiry_days'],
    config['preauthorized']
)

authenticator.login()

if st.session_state["authentication_status"]:
    conn = st.connection('gcs', type=FilesConnection)

    # Load the CSV data
    goal = conn.read('flux-storage/linkddd/Goal.csv',input_format='csv', ttl=600)
    project = conn.read('flux-storage/linkddd/Project.csv',input_format='csv', ttl=600)
    todo = conn.read('flux-storage/linkddd/TODO.csv',input_format='csv', ttl=600)

AHP = np.array([0.171213216,0.10970528,0.116156893,0.338304569,0.264620042])

def update_urgent(row):
    start_date = datetime.strptime(row['Start Date'], '%Y-%m-%d')
    urgent = (start_date-datetime.now()).days
    return urgent
    
def update_effort(row):
    start_date = datetime.strptime(row['Start Date'], '%Y-%m-%d')
    end_date = datetime.strptime(row['End Date'], '%Y-%m-%d')
    effort = (end_date-start_date).days
    return effort


goal_effort_map = goal.set_index('Goal')['Estimate Effort'].to_dict()
project_effort_map = project.set_index('Project Name')['Estimate Time Cost'].to_dict()

goal_impact_map = goal.set_index('Goal')['Impact'].to_dict()


# 使用 map 函数更新 'Goal Effort' 和 'Project Effort'
todo['Goal Effort'] = todo['Goal'].map(goal_effort_map)
todo['Project Effort'] = todo['Project'].map(project_effort_map)
todo['Total Effort'] = 3*todo['Project Effort'] + 1/3*todo['Goal Effort']
todo['Impact'] = todo['Goal'].map(goal_impact_map)
todo['Value'] = todo['Impact']* todo['Cost']/todo['Project Effort']
    
todo['Urgent'] = todo.apply(update_urgent, axis=1)
todo['Normal Urgent'] = -zscore(todo['Urgent'])
todo['Normal Cost'] = -zscore(todo['Cost'])
todo['Normal Effort'] = -zscore(todo['Total Effort'])
todo['Normal Value'] = zscore(todo['Value'])
todo['Normal Impact'] = zscore(todo['Impact'])


columns = ['Normal Urgent', 'Normal Cost', 'Normal Effort', 'Normal Value', 'Normal Impact']

todo['Score'] = todo[columns].apply(lambda row: np.sum(AHP * row), axis=1)
todo['Score'] = (todo['Score'] - todo['Score'].min()) / (todo['Score'].max() - todo['Score'].min())
todo.to_csv('Data/TODO.csv', index=False)


# Create a linear programming problem
prob = pulp.LpProblem("OptimizeTODO", pulp.LpMaximize)

# Create decision variables
todo['Start Date'] = pd.to_datetime(todo['Start Date']).dt.date

# Get today's date
today = datetime.today().date()

# Create decision variables
x = {i: (0 if todo['Start Date'][i] > today or todo['Status'][i] else pulp.LpVariable(f'x{i}', cat='Binary')) for i in range(len(todo))}
#x = {i: (0 if todo['Start Date'][i] > today else pulp.LpVariable(f'x{i}', cat='Binary')) for i in range(len(todo))}
# Objective function: maximize the total score
prob += pulp.lpSum(todo['Score'][i]*x[i] for i in range(len(todo)))

# Constraint: total cost should be less than or equal to 600
prob += pulp.lpSum(todo['Cost'][i]*x[i] for i in range(len(todo))) <= 360
# Create 'Task Main Part' column
todo['Task Main Part'] = todo['Task Name'].str.split().str[0]
# Add constraints: for each main part, at most one task can be selected
for _, group in todo.groupby('Task Main Part'):
    prob += pulp.lpSum(x[i] for i in group.index) <= 1
# Solve the problem
prob.solve()


# Initialize 'Optimal Solution' with zeros
todo['Optimal Solution'] = 0

# Assign optimal solutions
for i in x:
    todo.loc[i, 'Optimal Solution'] = pulp.value(x[i])


todo.to_csv('Data/TODO.csv', index=False)

time_category = json.load(open('Data/time_category.json','r'))


#-----精力管理-----#

with open ('Data/time_category.json', 'r') as f:
    category_data = json.load(f)
    
def calculate_score(c,weight):
    start_time = datetime.strptime(c["period"].split(" - ")[0], "%H:%M")
    end_time = datetime.strptime(c["period"].split(" - ")[1], "%H:%M")

    # 计算这是今天的第几个5分钟
    start_five_min_slot = (start_time.hour * 60 + start_time.minute) // 5
    end_five_min_slot = (end_time.hour * 60 + end_time.minute) // 5
    start_left_over = 5 - (start_time.hour * 60 + start_time.minute) % 5
    end_left_over = (end_time.hour * 60 + end_time.minute) % 5
    
    
    weight[start_five_min_slot] = start_left_over / 5
    weight[end_five_min_slot] = end_left_over / 5
    if start_five_min_slot == end_five_min_slot:
        weight[start_five_min_slot] = end_left_over / 5 - start_left_over / 5
    if end_five_min_slot - start_five_min_slot > 1:
        weight[start_five_min_slot+1:end_five_min_slot] = 1

    return start_left_over, end_left_over, start_five_min_slot, end_five_min_slot

data = pd.read_csv('Data/combined_data.csv')
unique_dates = data['date'].unique()
for date in unique_dates:
    date_data = data[data['date'] == date]
    for i in range(len(date_data)):
        c = date_data.iloc[i]



weight_ls = []
# 遍历每个独特的日期
for date in unique_dates:
    date_data = data[data['date'] == date]
    weight = np.zeros(288)
    for i in range(len(date_data)):
        c = date_data.iloc[i]
        for key in category_data.keys():
            if c['category'] in category_data[key]:
                if key == "Self Development" or key == "Main Work":
                    calculate_score(c,weight)
            else:
                pass 
    weight_ls.append(weight)


# 初始化总权重和计数器
total_weight = 0
count = 0

# 遍历权重列表
for i, weight in enumerate(weight_ls, start=1):
    # 将权重乘以系数（在这里，系数是日期的索引）
    total_weight += weight * i
    count += i

# 计算平均权重
average_weight = total_weight / count


def schedule_tasks(task_times, weight):
    # 初始化一个空的时间表
    schedule = [None] * len(weight)

    for task_index, task_time in enumerate(task_times):
        task_time_index = int(task_time/5)
        
        max_weight = 0
        max_weight_index = 0
        for i in range(len(weight)):
            if any(schedule[i:i+task_time_index]):
                continue
              
            total_weight = weight[i:i+task_time_index].sum()
            if total_weight > max_weight:
                max_weight = total_weight
                max_weight_index = i
                
        
        for i in range(max_weight_index, max_weight_index+task_time_index):
            if i >= len(weight):
                break
            print(i)
            schedule[i] = task_index+1
        

    return schedule

#-------- streamlit interface --------#

# 创建标题



if st.session_state["authentication_status"]:
    

    # 在侧边栏中添加一些控件
    with st.sidebar:
        option = st.sidebar.radio('Select an option', ['Today','Goal', 'Project', 'Todo','Time'])
        st.write(f'Welcom *{st.session_state["name"]}*')
        authenticator.logout()       
        
    if option == 'Today':
        st.title('Today')
        st.dataframe(todo[todo['Optimal Solution'] == 1])
        
        tasks = todo[todo['Optimal Solution'] == 1]['Cost']
        tasks_name = todo[todo['Optimal Solution'] == 1]['Task Name']
        schedule = schedule_tasks(tasks, average_weight)


        col1, col2= st.columns([1,3])
        with col1:
            st.subheader('Energy Heatmap')
            
            matrix = np.reshape(average_weight, (24, 12))

            # Create a heatmap with plotly
            fig = go.Figure(data=go.Heatmap(
                            z=matrix,
                            colorscale='Viridis'))  # Use a color scale with more colors

            fig.update_layout(
                autosize=False,
                width=300,  # Set the width to your desired value
                height = 600
            )

            fig.update_yaxes(autorange="reversed")  # Reverse the y-axis

            st.plotly_chart(fig)

        with col2:
            st.subheader('Schedule')
            data = []

            # 为每个任务创建一个字典
            for task_index in set(schedule):
                if task_index is not None:
                    task_time_indices = [i for i, x in enumerate(schedule) if x == task_index]
                    data.append(dict(Task=tasks_name.iloc[task_index-1], Start='2022-01-01 {:02d}:{:02d}:00'.format(min(task_time_indices)//12, (min(task_time_indices)%12)*5), Finish='2022-01-01 {:02d}:{:02d}:00'.format((max(task_time_indices)+1)//12, ((max(task_time_indices)+1)%12)*5)))
            # 创建一个Gantt图
            fig = ff.create_gantt(data, show_colorbar=True, group_tasks=True)
            fig.update_layout(width=1200)

            # 显示图形
            st.plotly_chart(fig)


        with st.form(key='time_today'):
            date = st.date_input(label='Enter Date')
            time_today = st.text_area(label='Enter Your Time Today')
            button = st.form_submit_button(label='Submit')
            
        if button:
            if time_today:
                lines = time_today.split('\n')
                for line in lines:
                    if line.strip():  # This will skip empty lines
                        category , time = line.split(':')
                        category = category.strip()
                        for cat in time_category.keys():
                            if category == cat:
                                time_entry = pd.read_csv('Data/Time.csv')
                                time_entry = time_entry.append({'Date':date,'Category':cat,'Subcategory':'','Time':time},ignore_index=True)
                                time_entry.to_csv('Data/Time.csv',index=False)
                            if category in time_category[cat]:
                                time_entry = pd.read_csv('Data/Time.csv')
                                time_entry = time_entry.append({'Date':date,'Category':cat,'Subcategory':category,'Time':time},ignore_index=True)
                                time_entry.to_csv('Data/Time.csv',index=False)

                        
        def new_subcateogry(target,sub):
            for key in time_category.keys():
                if target in key:
                    if sub not in time_category[key]:
                        time_category[key].append(sub)
            with open('Data/time_category.json','w') as f:
                json.dump(time_category,f)

        with st.form(key='new_category'):
            new_category = st.selectbox(label='Enter Category', options=time_category.keys())
            new_sub_category = st.text_input(label='Enter Sub Category')
            submit_button = st.form_submit_button(label='Add Category')
        
        if submit_button:
            new_subcateogry(new_category,new_sub_category)
        

    if option == 'Todo':
        st.title('Todo')
        col1, col2= st.columns(2)
        
        with col1:
            st.subheader('New Todo')
            with st.form(key='my_form'):
                task = st.text_input(label='Enter TODO Name')
                date = st.date_input(label='Enter Start Date')
                cost = st.number_input(label='Enter Cost', min_value=0, max_value=1000)
                type = st.selectbox(label='Enter type', options=['Event','Task','Routine'])
                goal = st.selectbox(label='Enter Goal', options=goal['Goal'])
                project = st.selectbox(label='Enter Project', options=project['Project Name'])
                submit_button = st.form_submit_button(label='Add TODO')

            # 当用户点击提交按钮时，将新的数据添加到 DataFrame 中
            if submit_button:
                new_data = {'Task Name': task, 'Cost': cost, 'Start Date': date, 'Type': type, 'Goal': goal, 'Project': project}
                todo = todo.append(new_data, ignore_index=True)

            # 将更新后的 DataFrame 保存回 CSV 文件
            todo.to_csv('Data/TODO.csv', index=False)
        with col2:
            goal = pd.read_csv('Data/Goal.csv')
            goal_selection = st.selectbox(label='Select Goal', options=goal['Goal'])
            st.subheader('Edit TODO')
            edit_data = st.data_editor(data=todo[todo['Goal'] == goal_selection])
            save_button = st.button(label='Save')
            
            if save_button:
                # Get the indices of the rows that match the selected goal
                indices = todo[todo['Goal'] == goal_selection].index

                # Set the indices of the edited DataFrame to match those in the original DataFrame
                edit_data.index = indices

                # Update the original DataFrame with the edited DataFrame
                todo.update(edit_data)

                # Save the updated DataFrame to the CSV file
                todo.to_csv('Data/TODO.csv', index=False)

    if option == 'Goal':
        st.title('Goal')
        col1, col2= st.columns(2)
        
        with col1:
            st.subheader('New Goal')
            with st.form(key='my_form'):
                goal_name = st.text_input(label='Enter Goal Name')
                date = st.date_input(label='Enter Start Date')
                cost = st.number_input(label='Enter Estimate Effort', min_value=0, max_value=50000)
                income_increase = st.number_input(label='Enter Income Increase', min_value=0, max_value=100000)
                income_gain = st.number_input(label='Enter Income Gain', min_value=0, max_value=100000)
                submit_button = st.form_submit_button(label='Add Goal')

            # 当用户点击提交按钮时，将新的数据添加到 DataFrame 中
            if submit_button:
                new_data = {'Goal': goal_name, 'Estimate Effort': cost, 'Start Date': date, 'Income Increase': income_increase, 'Income Gain': income_gain,'Impact':income_increase*4.3+income_gain}
                goal = goal.append(new_data, ignore_index=True)

            # 将更新后的 DataFrame 保存回 CSV 文件
            goal.to_csv('Data/Goal.csv', index=False)
        with col2:
            st.subheader('Edit Goal')
            edit_data = st.data_editor(data=goal)
            save_button = st.button(label='Save')
            
            if save_button:
                edit_data.to_csv('Data/Goal.csv', index=False)
            
    if option == 'Project':
        st.title('Project')
        col1, col2= st.columns(2)
        
        with col1:
            st.subheader('New Project')
            with st.form(key='my_form'):
                project_name = st.text_input(label='Enter Project Name')
                date = st.date_input(label='Enter Start Date')
                cost = st.number_input(label='Enter Estimate Effort', min_value=0, max_value=50000)
                goal_select = st.selectbox(label='Enter Goal', options=goal['Goal'])
                submit_button = st.form_submit_button(label='Add Goal')

            # 当用户点击提交按钮时，将新的数据添加到 DataFrame 中
            if submit_button:
                new_data = {'Project Name': project_name, 'Estimate Time Cost': cost, 'Start Date': date, 'Goal': goal_select}
                project = project.append(new_data, ignore_index=True)

            # 将更新后的 DataFrame 保存回 CSV 文件
            project.to_csv('Data/Project.csv', index=False)
        with col2:
            st.subheader('Edit Project')
            edit_data = st.data_editor(data=project)
            save_button = st.button(label='Save')
            
            if save_button:
                edit_data.to_csv('Data/Project.csv', index=False)
                
    if option == 'Time':
        # Assuming 'time' is a DataFrame with 'Date' and 'Time' columns
        # Assuming 'time' is a DataFrame with 'Date' and 'Time' columns
        st.title('Time')
        #time = pd.read_csv('Data/Time.csv')

        combined_data = pd.read_csv('Data/combined_data.csv')

        def calculate_duration(period):
            start, end = period.split(' - ')
            start_hour, start_minute = map(int, start.split(':'))
            end_hour, end_minute = map(int, end.split(':'))
            
            start_in_minutes = start_hour * 60 + start_minute
            end_in_minutes = end_hour * 60 + end_minute
            
            return end_in_minutes - start_in_minutes

        combined_data["duration"] = combined_data["period"].apply(calculate_duration)
        grouped_data = combined_data.groupby([pd.to_datetime(combined_data['date']).dt.date, 'category'])['duration'].sum()
        grouped_data = grouped_data.reset_index()
        
        def find_parent_category(category):
            for parent_category, child_categories in time_category.items():
                if category in child_categories:
                    return parent_category
                if category == parent_category:
                    return category
            return None

        grouped_data["parent_category"] = grouped_data["category"].apply(find_parent_category)
        time = grouped_data.rename(columns={"category": "Subcategory", "duration": "Time", "parent_category": "Category", "date": "Date"})
        with open('Data/time_category.json') as f:
            time_category = json.load(f)

        view = st.selectbox(label='Select View', options=['Weekly','Monthly','Anual'] )
        
        if view == 'Weekly':
            time['Date'] = pd.to_datetime(time['Date'])
            week = st.select_slider(label='Select Week', options=[i for i in range(1,53)])
            week_data = time[(time['Date'].dt.year == 2024) & (time['Date'].dt.week == week)]
            time = week_data
            
        if view == 'Monthly':
            time['Date'] = pd.to_datetime(time['Date'])
            month = st.select_slider(label='Select Month', options=[1,2,3,4,5,6,7,8,9,10,11,12])
            month_data = time[(time['Date'].dt.year == 2024) & (time['Date'].dt.month == month)]
            time = month_data
        if view == 'Anual':
            time['Date'] = pd.to_datetime(time['Date'])
            year = st.select_slider(label='Select Year', options=[2023,2024])
            year_data = time[(time['Date'].dt.year == year)]
            time = year_data


        col1,col2,col3,col4 = st.columns(4)
        
        with col1:
            # Convert 'Date' column to datetime type
            time['Date'] = pd.to_datetime(time['Date'])
            
            category_counts = time.groupby('Category')['Time'].sum()
            
            category = category_counts/time['Time'].sum()
            
            
            real_percentage = []
            try:
                real_percentage.append(category['Self Development']+category['Main Work'])
                real_percentage.append(category['Relationship'])
                real_percentage.append(category['Normal Life'])
                real_percentage.append(category['Break'])
                real_percentage.append(category['Health'])
            except KeyError:
                real_percentage.append(0)
            
            percentage = [0.45,0.2,0.15,0.135,0.065]
            
            average_relative_error = np.mean(np.abs(np.array(real_percentage)-np.array(percentage))/np.array(percentage))
            score = int(100 - 50*(average_relative_error))
            st.metric(label='Score', value=score)
            
            if view == 'Weekly':
                try:
                    st.subheader('Main Work ' + str(round(category_counts['Main Work']/60 + category_counts['Self Development']/60, 2)) + '/41')
                    st.subheader('Relationship ' + str(round(category_counts['Relationship']/60, 2)) + '/18')
                    st.subheader('Normal Life ' + str(round(category_counts['Normal Life']/60, 2)) + '/13')
                    st.subheader('Break ' + str(round(category_counts['Break']/60, 2)) + '/12.3')
                    st.subheader('Health ' + str(round(category_counts['Health']/60, 2)) + '/6')
                except KeyError:
                    pass
            
            if view == 'Monthly':
                try:
                    st.subheader('Main Work ' + str(round(category_counts['Main Work']/60 + category_counts['Self Development']/60, 2)) + '/175')
                    st.subheader('Relationship ' + str(round(category_counts['Relationship']/60, 2)) + '/77')
                    st.subheader('Normal Life ' + str(round(category_counts['Normal Life']/60, 2)) + '/56')
                    st.subheader('Break ' + str(round(category_counts['Break']/60, 2)) + '/52.7')
                    st.subheader('Health ' + str(round(category_counts['Health']/60, 2)) + '/38')
                except KeyError:
                    pass
        with col2:
            # Get the current date
            now = pd.Timestamp.now()

            # Filter the data for the current month
            if view == 'Weekly':
                current_month_data = time[(time['Date'].dt.year == 2024) & (time['Date'].dt.week == week)]
                
            if view == 'Monthly':
                current_month_data = time[(time['Date'].dt.year == 2024) & (time['Date'].dt.month == month)]
            if view == 'Anual':
                current_month_data = time[(time['Date'].dt.year == year)]

            # Calculate the total time for the current month
            print(current_month_data['Time'])
            total_time_current_month = current_month_data['Time'].sum()/60

            # Format the total time to two decimal places
            total_time_current_month_formatted = "{:.2f}".format(total_time_current_month)

            st.metric(label='Total Time', value=total_time_current_month_formatted)
            
        with col3:
            # Calculate the daily total time
            time['day_total'] = time.groupby('Date')['Time'].transform('sum')/60

            # Calculate the average daily time
            average_daily_time = time['day_total'].mean()

            # Format the average daily time to two decimal places
            average_daily_time_formatted = "{:.2f}".format(average_daily_time)

            st.metric(label='Average Daily Time', value=average_daily_time_formatted)
        
        with col4:
            # Filter the data for 'main work' and 'self development'
            effective_work = time[time['Category'].isin(['Main Work', 'Self Development'])]

            # Calculate the daily total effective time
            effective_work['day_total_effective'] = effective_work.groupby('Date')['Time'].transform('sum')/60

            # Calculate the average daily effective time
            average_daily_effective_time = effective_work['day_total_effective'].mean()

            # Format the average daily effective time to two decimal places
            average_daily_effective_time_formatted = "{:.2f}".format(average_daily_effective_time)

            st.metric(label='Average Daily Effective Time', value=average_daily_effective_time_formatted)
        
        
        col1, col2= st.columns(2)
        with col1:
            # Group the data by category and sum the time
            category_time = time.groupby('Category')['Time'].sum()

            # Create a Pie chart
            colors = ['#ADD8E6', '#B0E0E6', '#87CEFA', '#87CEEB', '#00BFFF']

            fig = go.Figure(data=go.Pie(labels=category_time.index, values=category_time.values, marker=dict(colors=colors)))

            st.plotly_chart(fig)
            # Group the data by category and subcategory and sum the time
            category_subcategory_time = time.groupby(['Category', 'Subcategory'])['Time'].sum().reset_index()

            fig = px.bar(category_subcategory_time, x='Time', y='Category', color='Subcategory', orientation='h', title='Total Time by Category and Subcategory', color_discrete_sequence=px.colors.sequential.Blues)

            st.plotly_chart(fig)
            
        with col2:
        # Group the data by date and sum the time
            # Group the data by date and category and sum the time
            time['Date'] = pd.to_datetime(time['Date'])
            time['day_total'] = time.groupby('Date')['Time'].transform('sum')/60

            fig = calplot(time, x = 'Date', y = 'day_total')

            st.plotly_chart(fig)
        
            daily_category_time = time.groupby([time['Date'].dt.date, 'Category'])['Time'].sum().reset_index()

            # Create a list of blue colors
            colors = ['#ADD8E6', '#B0E0E6', '#87CEFA', '#87CEEB', '#00BFFF']

            # Create a Bar chart for each category
            fig = go.Figure()
            for i, category in enumerate(daily_category_time['Category'].unique()):
                category_data = daily_category_time[daily_category_time['Category'] == category]
                fig.add_trace(go.Bar(x=category_data['Date'], y=category_data['Time'], name=category, marker_color=colors[i % len(colors)]))

            fig.update_layout(
                barmode='stack',
                title='Total Time per Day by Category',
                xaxis_title='Date',
                yaxis_title='Total Time (minutes)',
            )

            st.plotly_chart(fig)

elif st.session_state["authentication_status"] is False:
    st.error('Username/password is incorrect')
elif st.session_state["authentication_status"] is None:
    st.warning('Please enter your username and password')
st.sidebar.title('Sidebar')        