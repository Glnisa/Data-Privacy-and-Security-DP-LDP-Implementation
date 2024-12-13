import math, random
import numpy as np
import matplotlib.pyplot as plt
""" Globals """

DOMAIN = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]

""" Helpers """


def read_dataset(filename):
    """
        Reads the dataset with given filename.
    """

    result = []
    with open(filename, "r") as f:
        for line in f:
            result.append(int(line))
    return result


# You can define your own helper functions here. #

### HELPERS END ###

""" Functions to implement """


# GRR

# TODO: Implement this function!
def perturb_grr(val, epsilon):

    len_dom = len(DOMAIN)
    p = math.exp(epsilon) / (math.exp(epsilon) + len_dom - 1)
    

    domain= np.arange(start=0, stop=len_dom)

    rnd = random.random()
    if rnd <= p:
        return val
    else:
        return random.choice(domain[domain != val])


# TODO: Implement this function!
def estimate_grr(perturbed_values, epsilon):

    len_dom = len(DOMAIN)
    len_pert_vals= len(perturbed_values)

    ##parameters for grr
    p = math.exp(epsilon) / (math.exp(epsilon) + len_dom - 1)
    q = (1 - p) / (len_dom - 1)

    c = np.zeros(len_dom)

    for i in perturbed_values:
        c[i-1] +=1

    result = list()

    for i in c:
        estimate = (i-(len_pert_vals*q)) / (p-q)
        result.append(estimate)


    return result    

    
    


# TODO: Implement this function!
def grr_experiment(dataset, epsilon):
    true_vals = list()
    perturbed_vals = list()

    for user_val in dataset:

        true_vals.append(user_val)
        perturbed_vals.append(perturb_grr(user_val, epsilon))

    estimate_fre = estimate_grr(perturbed_vals, epsilon)

    c= np.zeros(len(estimate_fre))

    for r in true_vals:

        c[r-1] +=1

    c = c.tolist()

    return calculate_average_error(c, estimate_fre)        

def calculate_average_error(count, estimate_freq):

    result=0
    diff=[]

    for e1, e2 in zip(count, estimate_freq):
        diff.append(abs(e1-e2))

   
    result = np.sum(diff) / len(count)

    return result
    


# RAPPOR

# TODO: Implement this function!
def encode_rappor(val):

    len_dom = len(DOMAIN)

    bit_vector = np.zeros(len_dom)
    bit_vector[val - 1] = 1

    return bit_vector


# TODO: Implement this function!
def perturb_rappor(encoded_val, epsilon):
    ##probability p
    p = (math.exp(epsilon / 2)) / (math.exp(epsilon / 2) + 1)

    perturbed_bit_vec = encoded_val.copy()

    for bit_idx in range(len(encoded_val)):

        rnd = random.random()

        if rnd > p:
            if perturbed_bit_vec[bit_idx] == 1:
                perturbed_bit_vec[bit_idx] = 0
            else:
                perturbed_bit_vec[bit_idx] = 1

    perturbed_bit_vec = perturbed_bit_vec.tolist()
    return perturbed_bit_vec


# TODO: Implement this function!
def estimate_rappor(perturbed_values, epsilon):

    perturbed_bit_vec = np.array(perturbed_values)

    ##probabilities p and q
    N = len(perturbed_bit_vec)
    p = (math.exp(epsilon / 2)) / (math.exp(epsilon / 2) + 1)
    q = 1 / (math.exp(epsilon / 2) + 1)

    perturbed_sum_bit_vector = sum(perturbed_bit_vec)
    estimate_freq_vector = list()

    for sum_bit in perturbed_sum_bit_vector:

        num = sum_bit - (N * q)
        denom = p - q
        estimate_freq_vector.append(num / denom)


    return estimate_freq_vector


# TODO: Implement this function!
def rappor_experiment(dataset, epsilon):

    true_vals = list()
    perturbed_vals = list()

    for user_val in dataset:
        true_vals.append(user_val)
        perturbed_vals.append(perturb_rappor(encode_rappor(user_val), epsilon))

    est_freq = estimate_rappor(perturbed_vals, epsilon)

    c = np.zeros(len(est_freq))

    for r in true_vals:
        c[r - 1] += 1

    c = c.tolist()    

    return calculate_average_error(c, est_freq)


# OUE

# TODO: Implement this function!
def encode_oue(val):

    len_dom = len(DOMAIN)
    bit_vector = np.zeros(len_dom)
    bit_vector[val - 1] = 1

    return bit_vector


# TODO: Implement this function!
def perturb_oue(encoded_val, epsilon):

    ##probabilities p and q
    p = 1 / 2
    q = 1 / (math.exp(epsilon) + 1)

    perturbed_bit_vec = encoded_val.copy()

    for bit_idx in range(len(encoded_val)):

        if perturbed_bit_vec[bit_idx] == 0:
            rnd = random.random()
            if rnd <= q:
                perturbed_bit_vec[bit_idx] = 1
        else:
            rnd = random.random()
            if rnd > p:
                perturbed_bit_vec[bit_idx] = 0

    return perturbed_bit_vec


# TODO: Implement this function!
def estimate_oue(perturbed_values, epsilon):

    len_vec = len(perturbed_values)
    perturbed_sum_bit_vec = sum(perturbed_values)
    estimate_freq_vector = list()

    for sum_bit in perturbed_sum_bit_vec:

        num = 2 * ((math.exp(epsilon) + 1) * sum_bit - len_vec)
        denom = math.exp(epsilon) - 1

        estimate_freq_vector.append(num / denom)

    return estimate_freq_vector


# TODO: Implement this function!
def oue_experiment(dataset, epsilon):

    true_vals = list()
    perturbed_vals = list()

    for user_val in dataset:

        true_vals.append(user_val)
        bit_vec = encode_oue(user_val)
        perturbed_vals.append(perturb_oue(bit_vec, epsilon))

    estimate_freq = estimate_oue(perturbed_vals, epsilon)

    c = np.zeros(len(estimate_freq))

    for r in true_vals:
        c[r - 1] += 1

    c = c.tolist()

    return calculate_average_error(c, estimate_freq)


def main():
    dataset = read_dataset("msnbc-short-ldp.txt")

    print("GRR EXPERIMENT")
    for epsilon in [0.1, 0.5, 1.0, 2.0, 4.0, 6.0]:
        error = grr_experiment(dataset, epsilon)
        print("e={}, Error: {:.2f}".format(epsilon, error))

    print("*" * 50)

    print("RAPPOR EXPERIMENT")
    for epsilon in [0.1, 0.5, 1.0, 2.0, 4.0, 6.0]:
        error = rappor_experiment(dataset, epsilon)
        print("e={}, Error: {:.2f}".format(epsilon, error))

    print("*" * 50)

    print("OUE EXPERIMENT")
    for epsilon in [0.1, 0.5, 1.0, 2.0, 4.0, 6.0]:
        error = oue_experiment(dataset, epsilon)
        print("e={}, Error: {:.2f}".format(epsilon, error))


if __name__ == "__main__":
    main()

