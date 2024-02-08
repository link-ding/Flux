import json
import datetime
import pandas as pd

def update_urgent(row):
    start_date = datetime.strptime(row['Start Date'], '%Y-%m-%d')
    urgent = (start_date-datetime.now()).days
    return urgent


def preprocess_data(goal,project, todo):
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