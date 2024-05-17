# DisCo code repo

Code repo of work: Towards Harmonious Disentanglement and Collaboration between Tabular and Semantic Space for Recommendation (In submission for KDD 2024)

### Data

The raw data of our three dataset is at:
https://grouplens.org/datasets/movielens/1m/

https://cseweb.ucsd.edu/jmcauley/datasets.html

https://files.grouplens.org/datasets/mov

To generate processed data, run ```python data_preprocess.py``` in ```dissem_preprocess``` folder.

You can use arbitrary LMM model to generate semantic embedding, for ours, we use vicuna-13b

https://vicuna.lmsys.org/

### Baseline

run the ```run_base.py``` file in run folder, we provide three methods to choose different ways to embed user history.

* No history, use ```--model={model}```, for example ```python run_base.py --model=DeepFM```
* Average pooling on user history, use ```--model={model}Mean```, for example ```python run_base.py --model=DeepFMMean```. **This is our baseline model in main results table**
* Attention on user history, use ```--model={model}Att```, for example ```python run_base.py --model=DeepFMAtt```. 

### Our method

run the ```run_DisCo.py``` file in run folder, use ```--model=DisCo{model}``` to specify the backbone.