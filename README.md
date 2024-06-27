# Knowledge injection into the LCR-Rot-hop(-ont)++ Model for ABSC
This code enables the investigation of the effects of knowledge injection on Aspect-Based Sentiment Classification (ABSC). Specifically, it incorporates knowledge from a restaurant domain ontology into the state-of-the-art LCR-Rot-hop++ model, referred to as LCR-Rot-hop-ont++. Our research explores different regimes of knowledge injection: (1) during training, (2) during testing, and (3) during both training and testing. Additionally, we aim to identify the optimal amount of knowledge to be injected by experimenting with various hops through the ontology. A detailed guide for using the code is provided below.

## Setting up the programming environment 
  - Create a conda environment with Python 3.10
  - Run `pip install -r requirements.txt` in your terminal to download all necessary packages
  - Find the unprocessed SemEval-2015 and SemEval-2016 datasets for restaurant reviews and the domain ontology under `data/raw`  
 ``   
## LCR-Rot-hop-ont++ Model Implementation
**Step 1**: Create the contextual word embeddings to be used in the model by running `main_preprocess.py`.
- The data is first cleaned of implicit targets as they are not compatible with the model
- Create the necessary embeddings by specifying the year, phase (Train or Test), and number of hops through the ontology in `def main()`.
-  If you would like to do an ablation experiment (the effect of excluding soft positioning and/or the visibility matrix), set the default in line 117 to true. Notice that currently this is only set up for the testing regime.

**Step 2**: 
  
- Step 2: Run main_hyperparam.py, this code optimizes the hyperparameters and must be run for every specific task before training. Adapt the year and the amount of ontology hops 
          used  as needed (note specify the ontology hops used during training in line 80).


  
- Step 3: Run main_train.py, use the hyperparameters of the previous step and adapt the year and the amount of ontology hops used as needed.
- Step 4: Run main_validate.py, specify the --model "MODEL_PATH" when running and adapt the year and the amount of ontology hops used as needed. This code will provide the results 
          for  the performance of the model as output.
  
## Optional Files

## References for the data
- Schouten, K., Frˇasincar, F., and de Jong, F. (2017). Ontology-enhanced aspect-based sentiment analysis. In 17th International Conference on Web Engineering (ICWE 2017), volume 10360 of LNCS, pages 302–320. Springer.
- Pontiki, M., Galanis, D., Papageorgiou, H., Androutsopoulos, I., Manandhar, S., Al-Smadi, M., Al-Ayyoub, M., Zhao, Y., Qin, B., De Clercq, O., et al. (2016). Semeval-2016 task 5: Aspect based sentiment analysis. In 10th International Workshop on Semantic Evaluation (SemEval2016), pages 19–30. ACL.
- Pontiki, M., Galanis, D., Papageorgiou, H., Manandhar, S., and Androutsopoulos, I. (2015). Semeval-2015 task 12: Aspect-based sentiment analysis. In 9th International Workshop on Semantic Evaluation (SemEval 2015), pages 486–495. ACL.

## References for the code
- https://github.com/charlottevisser/LCR-Rot-hop-ont-plus-plus 
- https://github.com/wesselvanree/LCR-Rot-hop-ont-plus-plus/tree/c8d18b8b847a0872bd66d496e00d7586fdffb3db.
- https://github.com/Bjarten/early-stopping-pytorch/blob/master/pytorchtools.py
- Liu, W., Zhou, P., Zhao, Z., Wang, Z., Ju, Q., Deng, H., Wang, P.: K-BERT: Enabling language representation with knowledge graph. In: 34th AAAI Conference on Artificial Intelligence. vol. 34, pp. 2901–2908. AAAI Press (2020)
