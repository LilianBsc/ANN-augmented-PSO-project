import evopti.evobject as evobject
import evopti.algos as algos
import b_25
import tensorflow as tf
import numpy as np
import pandas as pd

ANNname = "model_champ2"
json_file = open("neural-networks//" + ANNname + '.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
ANN = tf.keras.models.model_from_json(loaded_model_json)
# load weights into new model
ANN.load_weights("neural-networks//" + ANNname + ".h5")
print("Loaded ANN from disk")

problem = evobject.Problem(lambda x : b_25.get_cost(x))

modelpso = evobject.PSO_model( N_pop=60, 
                                N_iter=500, 
                                n_variables=31, 
                                stopping_criteria="impBest", 
                                eps=0.001, 
                                c1=2.05, 
                                c2=2.05, 
                                w=0.7, 
                                p_turbulence=0.1,
                                boundaries=[0, 1]
                            )
# modelannpso = evobject.ANNPSO_model( N_pop=60, 
#                                N_iter=500, 
#                                n_variables=31, 
#                                stopping_criteria="impBest", 
#                                n_gen_stop=20,
#                                eps=0.001, 
#                                c1=2.05, 
#                                c2=2.05, 
#                                w=0.7, 
#                                p_turbulence=0.1,
#                                boundaries=[-1500, -500], # This boundaries have to be precised
#                                ANN=ANN,
#                                cap_mem=5,
#                             )

N = 10
results = np.zeros((6, N))

for i in range(N):
    memory1 = algos.pso(modelpso, problem)
    # memory2 = algos.ann_augmented_pso(modelannpso, problem)

    # results[0, i] = memory1["n iter"]
    # results[1, i] = memory1["n call ff"]
    # results[2, i] = memory1["output"][1]
    # results[3, i] = memory2["n iter"]
    # results[4, i] = memory2["n call ff"]
    # results[5, i] = memory2["output"][1]

# pd.DataFrame(results).to_csv("comppsoannpso.csv")