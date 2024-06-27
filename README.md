# Knowledge injection into the LCR-Rot-hop(-ont)++ Model for ABSC
This code enables the investigation of the effects of knowledge injection on Aspect-Based Sentiment Classification (ABSC). Specifically, it incorporates knowledge from a restaurant domain ontology into the state-of-the-art LCR-Rot-hop++ model, referred to as LCR-Rot-hop-ont++. Our research explores different regimes of knowledge injection: (1) during training, (2) during testing, and (3) during both training and testing. Additionally, we aim to identify the optimal amount of knowledge to be injected by experimenting with various hops through the ontology. A detailed guide for using the code is provided below.

## Setting up the programming environment 
  - Create a conda environment with Python 3.10
  - Run `pip install -r requirements.txt` in your terminal to download all necessary packages
  - Find the unprocessed SemEval-2015 and SemEval-2016 datasets for restaurant reviews and the domain ontology under `data/raw`  
  
## LCR-Rot-hop-ont++ Model Implementation
**Step 1**: Create the contextual word embeddings to be used in the model by running `main_preprocess.py`.
- The data is first cleaned of implicit targets as they are not compatible with the model
- Create the necessary embeddings by specifying the year, phase (Train or Test), and number of hops through the ontology in `def main()`.
-  If you would like to do an ablation experiment (the effect of excluding soft positioning and/or the visibility matrix), set the default in line 117 to true. Notice that currently this is only set up for the testing regime.

**Step 2**: Optimise model hyperparameters by running `main_hyperparam.py`. 
- Specify the number of hops to be used during the training phase (line 81), during testing (line 214), and the year of the dataset used (line 212). Note that if no knowledge is injected, the parameters _--val-ont-hops_ and _ont_hops_ should be set to None and the gamma parameter set to 0 (line 64).
- The current setting implements a batch size of 32 and a fixed number of 20 epochs per evaluation (i.e., a set of hyperparameters).
- The best hyperparameters are going to be saved in the `.json` file under `data/checkpoints/year_epochs20" where _year_ corresponds to the dataset you implement. If you would like to save the results then rename the file before running the code for a different model, otherwise it will be re-written.
- There is no stopping mechanism for this algorithm, you need to decide for yourself when to end it by pressing Ctrl + C.

**Step 3**: Run main_train.py to optimize model parameters. 
- Similar to Step 2, set the model specification in lines 23 to 31.
- In lines 42 to 45, manually insert the hyperparameters obtained in the previous step.
- The training process has a maximum of 100 epochs and stops early if no improvement occurs during 30 consecutive epochs.
- The best model will be saved under `data/models`

**Step 4**: Evaluate model performance by running `main_validate.py`.
- Set the model specification as before. The variable _--ont-hops_ in line 78 denotes the number of hops through the ontology to be used during testing.
- If knowledge is injected, change the value of gamma (line 82) to the one found in Step 2, otherwise set the default to None.
- To proceed with an ablation experiment set the default in line 90 to True.
- When running the file, you will get a prompt asking you to specify the model to be used. Do so by typing `python main_validate.py --model` plus the relative path to the model obtained in Step 3.
  
## Optional Files
-  `main_preprocess_count.py`: this code gives you an overview of the distribution of the aspects in the data
-  `main_hyperparam_colab.py`: the core idea and implementation are the same as in `main_hyperparam.py` with two adjustments. 
    1.  First, the code was made compatible with Google Colab
    2.  A stopping mechanism is implemented. The maximum number of evaluations is set to 100 and the same early stopping mechanism used for model training is implemented. That said, if validation accuracy does not improve after 30 consecutive evaluations the optimal hyperparameters are saved and the running is interrupted. 

## References for the data
- Schouten, K., Frˇasincar, F., and de Jong, F. (2017). Ontology-enhanced aspect-based sentiment analysis. In 17th International Conference on Web Engineering (ICWE 2017), volume 10360 of LNCS, pages 302–320. Springer.
- Pontiki, M., Galanis, D., Papageorgiou, H., Androutsopoulos, I., Manandhar, S., Al-Smadi, M., Al-Ayyoub, M., Zhao, Y., Qin, B., De Clercq, O., et al. (2016). Semeval-2016 task 5: Aspect-based sentiment analysis. In 10th International Workshop on Semantic Evaluation (SemEval2016), pages 19–30. ACL.
- Pontiki, M., Galanis, D., Papageorgiou, H., Manandhar, S., and Androutsopoulos, I. (2015). Semeval-2015 task 12: Aspect-based sentiment analysis. In 9th International Workshop on Semantic Evaluation (SemEval 2015), pages 486–495. ACL.

## References for the code
- https://github.com/charlottevisser/LCR-Rot-hop-ont-plus-plus 
- https://github.com/wesselvanree/LCR-Rot-hop-ont-plus-plus/tree/c8d18b8b847a0872bd66d496e00d7586fdffb3db.
- https://github.com/Bjarten/early-stopping-pytorch/blob/master/pytorchtools.py
- Liu, W., Zhou, P., Zhao, Z., Wang, Z., Ju, Q., Deng, H., Wang, P.: K-BERT: Enabling language representation with knowledge graph. In: 34th AAAI Conference on Artificial Intelligence. vol. 34, pp. 2901–2908. AAAI Press (2020)
