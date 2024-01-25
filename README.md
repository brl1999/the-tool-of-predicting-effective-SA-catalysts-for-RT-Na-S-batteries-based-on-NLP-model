# “Preferable single-atom catalysts enabled by natural language processing for room temperature Na-S batteries” Code Supplement 

This is the official code supplement of *Preferable single-atom catalysts enabled by natural language processing for room temperature Na-S batteries*.

The repository includes data from the abstracts of the papers used in this experiment, and code related to modeling and data analysis.



`abs_data/` contains relevant papers crawled from Elsevier.

`aug_emb_training.ipynb` is used to train our augmented embeddings.

`aug_emb_comparison_new.ipynb` helps to visualize experiments results.

---

**Setup**
Thanks to [1] outstanding work, we developed the code based on the MatSciBERT environment. Please refer to [this link](https://github.com/M3RG-IITD/MatSciBERT) first to configure the required environment.

In addition to above requirements, please install Seaborn in order to plot the charts.

`pip install seaborn`


- [1] [MatSciBERT: A materials domain language model for text mining and information extraction](https://www.nature.com/articles/s41524-022-00784-w)
---

**Citation**

If you find the code useful for your research, please consider citing our work:
*Preferable single-atom catalysts enabled by natural language processing for room temperature Na-S batteries*
