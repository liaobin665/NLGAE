# NLGAE
# Source of paper: NLGAE: a Graph Auto-Encoder Model Based on Improved Network Structure and Loss Function 

dependency:
 tensorflow: Compat. V1
 Scipy 
 pandas 
 numpy 
 xgboost 

each folder's function: 
 getDataX: extract each dataset X 
 Classfiy: classification performance experiments 
 getEmbeddings: input graph data, embedding 
 NLGAE: NLGAE model
 parameterTurnning: hyperparameter perturbation analysis 
 saveBestEmbedding: saving the best embedding output 
 XGBresult: Experiment with classifier, taking (XGB) as an example
