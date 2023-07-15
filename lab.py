from itertools import count
import evopti.evobject as obj
import evopti.test_functions as test_f
import tensorflow as tf
import evopti.algos as alg
import b_25
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd 
from mlxtend.plotting import heatmap
import matplotlib.pyplot as plt

# ANNname = "model_champ2"
# json_file = open(ANNname + '.json', 'r')
# loaded_model_json = json_file.read()
# json_file.close()
# ANN = tf.keras.models.model_from_json(loaded_model_json)
# # load weights into new model
# ANN.load_weights(ANNname + ".h5")
# print("Loaded ANN from disk")

N_samples = 1

Cn = 10**5
# x=[0 for _ in range(31)]
# y = b_25.ann_gwmodel(x, Cn, show_result=True, weights=[0,1])

n_scope = 4
w1_scope = np.linspace(0.1, 1., n_scope)
w2_scope = np.linspace(0.1, 1., n_scope)
w3 = 0

mat500 = np.zeros((len(w1_scope), len(w2_scope)))
mat1500 = np.zeros((len(w1_scope), len(w2_scope)))
matcouple = np.zeros((len(w1_scope), len(w2_scope)))
mat500_1500 = np.zeros((len(w1_scope), len(w2_scope)))
matoutput = np.zeros((len(w1_scope), len(w2_scope)))

for i in range(len(w1_scope)):
    for j in range(len(w2_scope)):
        w1 = w1_scope[i]
        w2 = w2_scope[j]
        problem = obj.Problem(lambda x : b_25.ann_gwmodel(x, Cn, weights=[w1,w2,w3]))
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
                                        boundaries=[-1500, -500], # This boundaries have to be precised
                                    )
        memory = alg.pso(pso_model, problem)
        output = memory["output"]
        mat500[i, j] = output[0].count(-500)
        mat1500[i, j] = output[0].count(-1500)
        mat500_1500[i, j] = mat500[i, j] + mat1500[i, j]
        # matcouple[i, j] = (mat500[i, j], mat1500[i, j])
        matoutput[i, j] = output[1]


heatmap(mat500_1500, figsize=(10, 10), cell_values=True)
plt.show()

# for i in range(len(w1_scope)):
#     print("w2 =", w1_scope[i])
#     print("-500 :", results[i][0].count(-500))
#     print("-1500 :", results[i][0].count(-1500))
#     print(results[i])

# for i in range(N_samples):
#     Cn = 10**5
#     problem = obj.Problem(lambda x : b_25.ann_gwmodel(x, Cn))
#     pso_model = obj.ANNPSO_model(   N_pop=60, 
#                                     N_iter=500, 
#                                     n_variables=31, 
#                                     stopping_criteria="impBest", 
#                                     n_gen_stop=20,
#                                     eps=0.001, 
#                                     c1=2.05, 
#                                     c2=2.05, 
#                                     w=0.7, 
#                                     p_turbulence=0.1,
#                                     boundaries=[-1500, -500], # This boundaries have to be precised
#                                     ANN=ANN,
#                                     cap_mem=5
#                                 )
#     memory1 = alg.ann_augmented_pso(pso_model, problem)
#     results[0, i] = memory1["n iter"]
#     results[1, i] = memory1["n call ff"]
#     results[2, i] = memory1["output"][1]

#     Cn = 10**5
#     problem = obj.Problem(lambda x : b_25.ann_gwmodel(x, Cn))
#     pso_model = obj.PSO_model(   N_pop=60, 
#                                     N_iter=500, 
#                                     n_variables=31, 
#                                     stopping_criteria="impBest", 
#                                     n_gen_stop=20,
#                                     eps=0.001, 
#                                     c1=2.05, 
#                                     c2=2.05, 
#                                     w=0.7, 
#                                     p_turbulence=0.1,
#                                     boundaries=[-1500, -500], # This boundaries have to be precised
#                                 )
#     memory2 = alg.pso(pso_model, problem)
#     results[3, i] = memory2["n iter"]
#     results[4, i] = memory2["n call ff"]
#     results[5, i] = memory2["output"][1]

#     pd.DataFrame(results).to_csv("results-GW.csv")

