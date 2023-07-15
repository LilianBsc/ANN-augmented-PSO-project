import evopti.lab as lab
import evopti.test_functions as tf
import evopti.algos as alg
import evopti.evobject as obj
import b_25
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd 
import random

def distance(v1, v2):
    n_v = len(v1)
    return sum([(v1[i]-v2[i])**2 for i in range(n_v)])**.5

def centroid(cloud):
    centroid = [0 for _ in range(len(cloud[0]))]
    for p in cloud:
        for i in range(len(p)):
            centroid[i] += p[i]
    for i in range(len(centroid)):
        centroid[i] = centroid[i]/len(cloud)
    return centroid

def angle(v1, v2):
    inner = np.inner(v1, v2)
    norms = np.linalg.norm(v1) * np.linalg.norm(v2)

    cos = inner / norms
    rad = np.arccos(np.clip(cos, -1.0, 1.0))
    return rad

def build_dataset(dataset_filename, inject_positives=False):
    Cn = 10**5
    # _, _, memory = lab.test2(10, 0.8, 0., lambda x : b_25.ann_gwmodel(x, Cn),
    #     31, 1, stopping_criteria="impBest", 
    #     eps=0.001, N_iter=500, N_pop=60,
    #     constriction=True, cube=[-2500, -500], w=0.7)

    # _, _, memory = lab.test2(10, 0.8, 0., tf.ackley,
    #                         5, 1, stopping_criteria="impBest", 
    #                         eps=0.001, N_iter=600, N_pop=60,
    #                         constriction=True, cube=[-500, 500], w=0.7)

    problem = obj.Problem(lambda x : b_25.ann_gwmodel(x, Cn))
    pso_model = obj.PSO_model(  N_pop=60, 
                                N_iter=500, 
                                n_variables=31, 
                                stopping_criteria="impBest", 
                                eps=0.001, 
                                c1=2.05, 
                                c2=2.05, 
                                w=0.7, 
                                boundaries=[-2500, -500] # This boundaries have to be precised
                            )
    memory = alg.pso(pso_model, problem)

    positions = memory["positions"]
    pbests = memory["pbests"]
    gbests = memory["gbests"]
    pbcosts = memory["pbests costs"]

    n_gen = positions.shape[0]
    n_pop = positions.shape[1]

    # print(positions.shape)
    # print(pbests.shape)
    # print(gbests.shape)
    # print(pbcosts.shape)

    data1 = np.zeros((n_pop, n_gen-1))
    data2 = np.zeros((n_pop, n_gen-1))
    data3 = np.zeros((n_pop, n_gen-1))
    data4 = np.zeros((n_pop, n_gen-1))
    data5 = np.delete(pbcosts, -1, axis=1)
    data6 = np.zeros((n_pop, n_gen-1))


    for p in range(n_pop):
        for gen in range(n_gen-2):
            centroid = positions[gen].mean(axis=0)

            data1[p, gen+1] = distance(positions[gen, p], positions[gen+1, p])
            data2[p, gen+1] = distance(pbests[gen, p], pbests[gen+1, p])
            data3[p, gen] = distance(positions[gen, p], gbests[gen])
            data4[p, gen] = distance(pbests[gen, p], gbests[gen])
            data6[p, gen] = distance(centroid, positions[gen, p])
        data3[p, n_gen-2] = distance(positions[n_gen-1, p], gbests[n_gen-1])
        data4[p, n_gen-2] = distance(pbests[n_gen-1, p], gbests[n_gen-1])
        data6[p, gen] = distance(positions[n_gen-1].mean(axis=0), positions[n_gen-1, p])

    vels = memory["velocities"]
    gbests = memory["gbests"]
    pbchanges = memory["pbest changes"]

    gbests_costs = []
    for gen in range(memory["n iter"]):
        gbests_costs.append(memory[gen]["gbest cost"])

    pd.DataFrame(data1).to_csv("data1.csv")
    pd.DataFrame(data2).to_csv("data2.csv")
    pd.DataFrame(data3).to_csv("data3.csv")
    pd.DataFrame(data4).to_csv("data4.csv")
    pd.DataFrame(data5).to_csv("data5.csv")
    # plt.plot(gbests_costs)
    # plt.show()

    datalist = [data1, data2, data3, data4, data5]
    cap_mem = 5
    dataset = np.empty((1, (len(datalist)-1)*cap_mem + 1))
    sample_rate = 0.4

    row = np.array([])
    parts = [i for i in range(n_pop)]
    gens = [i for i in range(cap_mem, n_gen-1)]
    couples = [(i, j) for i in parts for j in gens]

    if inject_positives:
        for couple in couples:
            i, j = couple
            if pbchanges[i, j] == 1 or data3[i, j] == 0:
                for data in datalist:
                    row = np.concatenate((row, data[i, j - cap_mem : j-1]), axis=0)
                row = np.concatenate((row, [pbchanges[i, j]]), axis=0)
                row[-1] = 1
                if row.size == (cap_mem-1)*len(datalist)+1:
                    dataset = np.append(dataset, [row], axis=0)
                row = np.array([])
    else:
        while len(dataset) < n_pop*(n_gen-10)*sample_rate and len(couples)>0:
            couple = random.randint(0,len(couples)-1)
            i, j = couples[couple]

            for data in datalist:
                row = np.concatenate((row, data[i, j - cap_mem : j-1]), axis=0)
            row = np.concatenate((row, [pbchanges[i, j]]), axis=0)
            if row[-1] == 0:
                if data3[i, j] == 0:
                    row[-1] = 1
            if row.size == (cap_mem-1)*len(datalist)+1:
                dataset = np.append(dataset, [row], axis=0)
            # else:
            #     print("i, j = ", i, j)
            row = np.array([])
            couples.pop(couple)

    dataset = np.delete(dataset, (0), axis=0)
    print("dataset size", dataset.shape)
    pd.DataFrame(dataset).to_csv(dataset_filename, mode="a", header=False, index=False)

    pd.DataFrame(pbchanges).to_csv("pbest-changes.csv")


def build_simple_dataset(filename, inject_positives=False):
    Cn = 10**5
    # _, _, memory = lab.test2(10, 0.8, 0., lambda x : b_25.ann_gwmodel(x, Cn),
    #     31, 1, stopping_criteria="impBest", 
    #     eps=0.001, N_iter=500, N_pop=60,
    #     constriction=True, cube=[-2500, -500], w=0.7)

    # _, _, memory = lab.test2(10, 0.8, 0., tf.ackley,
    #                         5, 1, stopping_criteria="impBest", 
    #                         eps=0.001, N_iter=600, N_pop=60,
    #                         constriction=True, cube=[-500, 500], w=0.7)

    problem = obj.Problem(tf.ackley)
    pso_model = obj.PSO_model(  N_pop=60, 
                                N_iter=500, 
                                n_variables=31, 
                                stopping_criteria="impBest", 
                                n_gen_stop=20,
                                eps=0.001, 
                                c1=2.05, 
                                c2=2.05, 
                                w=0.7, 
                                p_turbulence=0.1,
                                boundaries=[500, 500] # This boundaries have to be precised
                            )
    memory = alg.pso(pso_model, problem)

    positions = memory["positions"]
    pbests = memory["pbests"]
    gbests = memory["gbests"]
    pbcosts = memory["pbests costs"]

    n_gen = positions.shape[0]
    n_pop = positions.shape[1]

    # print(positions.shape)
    # print(pbests.shape)
    # print(gbests.shape)
    # print(pbcosts.shape)


    data1 = np.zeros((n_pop, n_gen-1))
    data2 = np.zeros((n_pop, n_gen-1))
    
    for p in range(n_pop):
        for gen in range(1, n_gen-1):
            if distance(positions[gen, p], gbests[gen]) == 0:
                data1[p, gen] = 1
            if distance(pbests[gen-1, p], pbests[gen, p]) != 0:
                data2[p, gen] = 1

    pbchanges = memory["pbest changes"]

    datalist = [data1, data2]
    cap_mem = 5
    dataset = np.empty((1, len(datalist)*cap_mem + 1))
    sample_rate = 0.4

    row = np.array([])
    parts = [i for i in range(n_pop)]
    gens = [i for i in range(cap_mem, n_gen-1)]
    couples = [(i, j) for i in parts for j in gens]

    while len(dataset) < n_pop*(n_gen-cap_mem)*sample_rate and len(couples)>0:
        i_couple = random.randint(0,len(couples)-1)
        i, j = couples[i_couple]

        for data in datalist:
            row = np.concatenate((row, data[i, j - cap_mem : j]), axis=0)
            # print(data[i, j - cap_mem : j])
            # print(f"({i}, {j})=", data[i, j])
            # print(f"({i}, {j-1})=", data[i, j-1])
            # print(f"({i}, {j-cap_mem})=", data[i, j - cap_mem])
        row = np.concatenate((row, [pbchanges[i, j]]), axis=0)
        if row[-1] == 0:
            if data1[i, j] == 1:
                row[-1] = 1
        if row.size == (cap_mem)*len(datalist)+1:
            dataset = np.append(dataset, [row], axis=0)
        # else:
        #     print("i, j = ", i, j)
        row = np.array([])
        couples.pop(i_couple)

    dataset = np.delete(dataset, (0), axis=0)

    print("dataset size", dataset.shape)
    pd.DataFrame(dataset).to_csv(filename, mode="a", header=False, index=False)

    pd.DataFrame(pbchanges).to_csv("pbest-changes.csv")

for _ in range(200):
    build_simple_dataset("ackley_dataset_test_simple.csv", inject_positives=False)