# ANN-augmented-PSO-project

Remembered paramters to train ANN :
- PBest change (0, 1)
- Gbest (0, 1)
- Distance between particule and Gbest / same Pbest
- Position in the swarm (To find)
- Velocity
- Cost for last generations
- Neiberhood density
- Trajectory (formula to compute if it is a straight line or zig-zags)

All parameters, you can take for the n past generations

Before implementing ANN : make statistics on each features to detect outliers or else, and CROSSCORELATION (pandas dataframe)
Once you have your model, use SHAP Value to understand it better and see the influence of each parameters in the classification (links: https://shap.readthedocs.io/en/latest/)
ANN is maybe not the best option, try this instead : instead of just 0, 1 classifications find a way to give rates (0 - 10) to the particules, then it becomes a 10 part classification problem and it will be more accurate.
Try using LGBM (simple and best), or GA / PSO regression with the Tau-b correlation (call me if you need).