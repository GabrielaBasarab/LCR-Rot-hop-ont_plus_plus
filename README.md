# Knowledge injection into the LCR-Rot-hop(-ont)++ model for ABSC
This code allows to investigate the effect of knowledge injection on ABSC. Specifically, it injects knowledge from a restaurant domain ontology into the state-of-the-art LCR-Rot-hop++ model, called LCR-Rot-hop-ont++. In our research, we investigated different knowledge injection regimes: (1) training, (2) testing, and (3) testing and training. Moreover, we tried to optimize the amount of knowledge to be 





This code can be used to train and validate LCR-Rot-hop(-ont)++ models. The model is a neural network used for Aspect-Based Sentiment Classification and
allows for injection knowledge from an Ontology. 

## Before running the code
- Set up environment:
  - Create a conda environment with Python 3.10
  - run `pip install -r requirements.txt` in your terminal 
- Set up data
  - the data and ontologies can be found at `data/raw`  
  - for simplicity the SemEval 2014 laptop dataset is named "ABSA14_Restaurants_...."

## Training and validating process
Note that this process works for the 2015 and 2016 dataset, in the case of the 2014 datasets some adaptations have to be made to the code. These adaptations are explained below. 

- Step 1: Run main_preprocess.py, adapt the year and the amount of ontology hops used as needed.
- Step 2: Run main_hyperparam.py, this code optimizes the hyperparameters and must be run for every specific task before training. Adapt the year and the amount of ontology hops 
          used  as needed (note specify the ontology hops used during training in line 80).
- Step 3: Run main_train.py, use the hyperparameters of the previous step and adapt the year and the amount of ontology hops used as needed.
- Step 4: Run main_validate.py, specify the --model "MODEL_PATH" when running and adapt the year and the amount of ontology hops used as needed. This code will provide the results 
          for  the performance of the model as output. 


## References for the data
- 


## References for the code
- https://github.com/charlottevisser/LCR-Rot-hop-ont-plus-plus 
- https://github.com/wesselvanree/LCR-Rot-hop-ont-plus-plus/tree/c8d18b8b847a0872bd66d496e00d7586fdffb3db.
- https://github.com/Bjarten/early-stopping-pytorch/blob/master/pytorchtools.py
- Liu, W., Zhou, P., Zhao, Z., Wang, Z., Ju, Q., Deng, H., Wang, P.: K-BERT: Enabling language representation with knowledge graph. In: 34th AAAI Conference on Artificial Intelligence. vol. 34, pp. 2901–2908. AAAI Press (2020)
