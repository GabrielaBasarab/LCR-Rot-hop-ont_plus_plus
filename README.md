# Knowledge injection into the LCR-Rot-hop(-ont)++ Model for ABSC
This code enables the investigation of the effects of knowledge injection on Aspect-Based Sentiment Classification (ABSC). Specifically, it incorporates knowledge from a restaurant domain ontology into the state-of-the-art LCR-Rot-hop++ model, referred to as LCR-Rot-hop-ont++. Our research explores different regimes of knowledge injection: (1) during training, (2) during testing, and (3) during both training and testing. Additionally, we aim to identify the optimal amount of knowledge to be injected by experimenting with various hops through the ontology. A detailed guide for using the code is provided below.

## Setting up the programming environment 
  - Create a conda environment with Python 3.10
  - Run `pip install -r requirements.txt` in your terminal to download all necessary packages
  - Locate the unprocessed SemEval-2015 and SemEval-2016 datasets for restaurant reviews and the domain ontology in the data/raw directory.
  
## LCR-Rot-hop-ont++ Model Implementation
**Step 1**: Create the contextual word embeddings for the model by running `main_preprocess.py`.
- The data is first cleaned of implicit targets as they are not compatible with the model
- Create the necessary embeddings by specifying the year, phase (Train or Test), and number of hops through the ontology in `def main()`.
- For ablation experiments (excluding soft positioning and/or the visibility matrix), set the default to true in line 117. Note that this is only set up for the testing regime.

**Step 2**: Optimise model hyperparameters by running `main_hyperparam.py`. 
- Specify the number of hops during training (line 81) and testing (line 214), and the year of the dataset (line 212).
- If no knowledge is injected, set _--val-ont-hops_ and _ont_hops_ to _None_, and _gamma_ to 0 (line 64).
- The current setting implements a batch size of 32 and a fixed number of 20 epochs per evaluation.
- The best hyperparameters are saved in a `.json` file under `data/checkpoints/year_epochs20", where _year_ corresponds to the dataset you implement. Rename the file before running the code for a different model to avoid overriding.
- There is no stopping mechanism; terminate the process manually using Ctrl + C when necessary.

**Step 3**: Run main_train.py to optimize model parameters. 
- Similar to Step 2, set the model specification in lines 23 to 31.
- Manually insert the hyperparameters obtained in the previous step in lines 42 to 45.
- The training process has a maximum of 100 epochs and employs early stopping if no improvement occurs over 30 consecutive epochs.
- The best model is saved under `data/models`

**Step 4**: Evaluate model performance by running `main_validate.py`.
- Set the model specification as before. The _--ont-hops_ variable in line 78 denotes the number of ontology hops used during testing.
- If knowledge is injected, set _gamma_ (line 82) to the value found in Step 2; otherwise, set it to _None_.
- For ablation experiments, set the default in line 90 to True.
- Run the file and specify the model by typing `python main_validate.py --model` followed by the relative path to the model obtained in Step 3, in the terminal.
  
## Optional Files
-  `main_preprocess_count.py`: provides an overview of the distribution of the aspects in the data
-  `main_hyperparam_colab.py`: the core idea and implementation are the same as in `main_hyperparam.py` with two adjustments. 
    1. The code was made compatible with Google Colab
    2. Includes a stopping mechanism with a maximum of 100 evaluations and early stopping if validation accuracy does not improve after 30 consecutive evaluations. The optimal hyperparameters are then saved, and the process is interrupted.
   
## References for the data
- Schouten, K., Frˇasincar, F., and de Jong, F. (2017). Ontology-enhanced aspect-based sentiment analysis. In 17th International Conference on Web Engineering (ICWE 2017), volume 10360 of LNCS, pages 302–320. Springer.
- Pontiki, M., Galanis, D., Papageorgiou, H., Androutsopoulos, I., Manandhar, S., Al-Smadi, M., Al-Ayyoub, M., Zhao, Y., Qin, B., De Clercq, O., et al. (2016). Semeval-2016 task 5: Aspect-based sentiment analysis. In 10th International Workshop on Semantic Evaluation (SemEval2016), pages 19–30. ACL.
- Pontiki, M., Galanis, D., Papageorgiou, H., Manandhar, S., and Androutsopoulos, I. (2015). Semeval-2015 task 12: Aspect-based sentiment analysis. In 9th International Workshop on Semantic Evaluation (SemEval 2015), pages 486–495. ACL.

## References for the code
- https://github.com/charlottevisser/LCR-Rot-hop-ont-plus-plus 
- https://github.com/wesselvanree/LCR-Rot-hop-ont-plus-plus/tree/c8d18b8b847a0872bd66d496e00d7586fdffb3db.
- https://github.com/Bjarten/early-stopping-pytorch/blob/master/pytorchtools.py
- Liu, W., Zhou, P., Zhao, Z., Wang, Z., Ju, Q., Deng, H., Wang, P.: K-BERT: Enabling language representation with knowledge graph. In: 34th AAAI Conference on Artificial Intelligence. vol. 34, pp. 2901–2908. AAAI Press (2020)
