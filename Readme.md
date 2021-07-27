Deep Performance Factors Analysis for Knowledge Tracing
---
This is the implementation of model in the paper: [Deep Performance Factors Analysis for Knowledge Tracing
]()

The works will be presented at AIED 2021:

Pu, S., Converse, G., & Huang, Y. (2021, June). Deep Performance Factors Analysis for Knowledge Tracing. In International Conference on Artificial Intelligence in Education (pp. 331-341). Springer, Cham.

To cite this work:
```
@inproceedings{pu2021deep,
  title={Deep Performance Factors Analysis for Knowledge Tracing},
  author={Pu, Shi and Converse, Geoffrey and Huang, Yuchi},
  booktitle={International Conference on Artificial Intelligence in Education},
  pages={331--341},
  year={2021},
  organization={Springer}
}
```


Dependencies
---
The code is developed and tested in `python 3.7.10`

to install dependencies: 
`pip install -r requirements.txt`

Run
---
use default configurations:  
`python main.py --dataset assist2017`  
`python main.py --dataset stat2011`  
`python main.py --dataset syn5`  
`python main.py --dataset nips2020`

Example
---
see the notebook in `doc/dpfa_example.ipynb` for how to run the model in google colab.
see the notebook in `doc/dpfa_benchmark.ipynb` for how to load the model as a `tensowflow.keras` model in google colab.

Data
---
The data used in the study is availabe in the `data` folder except for the [NeurIPS 2020 Education Challenge](https://competitions.codalab.org/competitions/25449)
dataset, which is too large to load. You could either use the link to download and partiion the data yourself, or send an email to the first author to request the data.


Issues
----
Please use the GitHub issue section to raise issues. 

Contact
---
For questions, feel free to either create a new issue or contact the first author: `scott.pu.pennstate@gmail.com`
