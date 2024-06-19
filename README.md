# DisCo

This is the implementation of "DisCo: Towards Harmonious Disentanglement and Collaboration between Tabular and Semantic Space for Recommendation".

### Data

The raw data of the three dataset is available at:

https://grouplens.org/datasets/movielens/1m/

https://cseweb.ucsd.edu/jmcauley/datasets.html

https://files.grouplens.org/datasets/mov

To generate the processed data, run ```python data_preprocess.py``` in ```dissem_preprocess``` folder.

You can use an arbitrary LLM model to generate semantic embedding, for ours, we use vicuna-13b:

https://vicuna.lmsys.org/

### Baseline

run the ```run_base.py``` file in run folder, we provide three methods to choose different ways to embed user history.

* No history, use ```--model={model}```, for example ```python run_base.py --model=DeepFM```
* Average pooling on user history, use ```--model={model}Mean```, for example ```python run_base.py --model=DeepFMMean```. **This is our baseline model in main results table**
* Attention on user history, use ```--model={model}Att```, for example ```python run_base.py --model=DeepFMAtt```. 

### Our method

run the ```run_DisCo.py``` file in run folder, use ```--model=DisCo{model}``` to specify the backbone.

### Citation

If you find this repo useful, please cite our paper.

@misc{du2024disco,

      title={DisCo: Towards Harmonious Disentanglement and Collaboration between Tabular and Semantic Space for Recommendation}, 
      
      author={Kounianhua Du and Jizheng Chen and Jianghao Lin and Yunjia Xi and Hangyu Wang and Xinyi Dai and Bo Chen and Ruiming Tang and Weinan Zhang},
      
      year={2024},
      
      eprint={2406.00011},
      
      archivePrefix={arXiv},
      
      primaryClass={cs.IR}
}
