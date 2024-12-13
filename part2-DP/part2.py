import numpy as np
import math
import csv
import matplotlib.pyplot as plt



''' Functions to implement '''

# TODO: Implement this function!
def read_dataset(file_path):

    ds = []
    ds_row = [] 
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        next(reader)
        for row in reader:
            for e in row:
                ds_row.append(e)
            
            ds_row_duplicate= ds_row.copy()
            ds.append(ds_row_duplicate)
            ds_row.clear()
            
    return ds


# TODO: Implement this function!
def get_histogram(dataset, state='TX', year='2020'):

    ds = dataset.copy()
    ds_2 = []
    ##list = [0]*12
    list=[]

    for row in ds:

        if row[1] == state:
            if row[0].find(year) > - 1:
                ds_2.append(row)

    for e in ds_2:
        ##month = int(e[0].split('-')[1])
        ##list[month - 1] += int(e[4])

        a=int(e[4])
        list.append(a) 
    
    ##draw_histogram(list, state, year)
    return list

'''Function to draw histogram'''
def draw_histogram(monthly_counts, state='TX', year='2020'):
    
    months = [f"{i:02d}" for i in range(1, 13)]
    
    
    plt.figure(figsize=(8, 6)) 
    plt.bar(months, monthly_counts, color='steelblue', edgecolor='black') 
    plt.title(f"Positive Test Cases for State {state} in year {year}")
    plt.xlabel("Month")
    plt.ylabel("Cases")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout() 
    plt.show()


# TODO: Implement this function!
def get_dp_histogram(dataset, state, year, epsilon, N):

    ds = dataset.copy()
    ds_2 = []
    list = []

    for row in ds:

        if row[1] == state:
            if row[0].find(year) > - 1:
                ds_2.append(row)

    for e in ds_2:
        ##defining noise with scale parameter= N/epsilon
        noise_to_add= np.random.laplace(0, N/epsilon)
        a=int(e[4])
        a += noise_to_add
        list.append(a)  


    ##draw_histogram(list, state, year)
    return list


# TODO: Implement this function!
def calculate_average_error(actual_hist, noisy_hist):

    result=0
    diff=[]

    for e1, e2 in zip(actual_hist, noisy_hist):
        diff.append(abs(e1-e2))

   
    result = np.sum(diff) / len(actual_hist)
    


    return result


# TODO: Implement this function!
def epsilon_experiment(dataset, state, year, eps_values, N):
    result = []

    for eps in eps_values:

        result_2 = []

        for _ in range(10):

            actual_hist = get_histogram(dataset, state, year)
            noisy_hist = get_dp_histogram(dataset, state, year, eps, N)
            
            result_2.append(calculate_average_error(actual_hist, noisy_hist))

        result.append(np.average(result_2))    
        
    return result


# TODO: Implement this function!
def N_experiment(dataset, state, year, epsilon, N_values):

    result = []

    for i in N_values:

        result_2 = []

        for _ in range(10):

            actual_hist = get_histogram(dataset, state, year)

            noisy_hist = get_dp_histogram(dataset, state, year, epsilon, i)

            result_2.append(calculate_average_error(actual_hist, noisy_hist))

        result.append(np.average(result_2))    

    return result


# FUNCTIONS FOR LAPLACE END #
# FUNCTIONS FOR EXPONENTIAL START #


# TODO: Implement this function!
def max_deaths_exponential(dataset, state, year, epsilon):

    ds = dataset.copy()
    ds_2 = []

    deaths = []
    p_list = []
    p_list_2 = []

    ##get values by looking to given state and year
    for row in ds:
        if row[1] == state:
            if row[0].find(year) > -1:
                ds_2.append(row)

    ##get deaths            
    for e in ds_2:

        a= int(e[2])
        deaths.append(a)

    for death in deaths:

        num = math.exp((epsilon*death)/2)
        p_list.append(num)    

    denom = np.sum(p_list)

    for p in p_list:
        p_list_2.append(p/denom)

    months = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']

    return_result = np.random.choice(months, p=p_list_2) 

    return return_result


# TODO: Implement this function!
def exponential_experiment(dataset, state, year, epsilon_list):

    result = []

    ds = dataset.copy()

    deaths = []
    ds_2= []

    for row in ds:

        if row[1] == state:
            if row[0].find(year) > -1:
                ds_2.append(row)

    for e in ds_2:

        a= int(e[2])
        deaths.append(a)   

    months = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']

    max_d = np.max(deaths)

    max_index = deaths.index(max_d)  

    ans = months[max_index]

    for eps in epsilon_list:

        c= 0
        for i in range(10000):
            if ans == max_deaths_exponential(dataset, state, year, eps):
                c += 1
        result.append(c/100)
           
    return result



# FUNCTIONS TO IMPLEMENT END #


def main():
    filename = "covid19-states-history.csv"
    dataset = read_dataset(filename)
    
    state = "TX"
    year = "2020"

    print("**** LAPLACE EXPERIMENT RESULTS ****")
    eps_values = [0.0001, 0.001, 0.005, 0.01, 0.05, 0.1, 1.0]
    error_avg = epsilon_experiment(dataset, state, year, eps_values, 2)
    for i in range(len(eps_values)):
        print("eps = ", eps_values[i], " error = ", error_avg[i])


    print("**** N EXPERIMENT RESULTS ****")
    N_values = [1, 2, 4, 8]
    error_avg = N_experiment(dataset, state, year, 0.5, N_values)
    for i in range(len(N_values)):
        print("N = ", N_values[i], " error = ", error_avg[i])

    state = "WY"
    year = "2020"

    print("**** EXPONENTIAL EXPERIMENT RESULTS ****")
    eps_values = [0.0001, 0.001, 0.01, 0.05, 0.1, 1.0]
    exponential_experiment_result = exponential_experiment(dataset, state, year, eps_values)
    for i in range(len(eps_values)):
        print("eps = ", eps_values[i], " accuracy = ", exponential_experiment_result[i])



if __name__ == "__main__":
    main()
