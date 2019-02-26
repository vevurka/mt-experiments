# Experiments

Experiments are divided into two directories:

1. `gdf`
2. `queue_imbalance`

In each directory for each experiment `experiment_name` there is:
1. one executable python file (`[experiment_name].py` or `run_[experiment_name].py`) 
2. one directory for the results (`res_[experiment_name]`)
3. one jupyter notebook visualizing the results (`[experiment_name].ipynb`)

To run experiments you need data in `data` directory:
* `prepared` - data extracted from raw LOB
* `data_gdf` - data after applying GDF

The data is prepared from data from raw LOB which you should put in `data/LOB` in format:
`OrderBookSnapshots_[stock]_[month][day].csv`, for instance `OrderBookSnapshots_9061_1016.csv`.

To prepare data run `$ python prepare_data.py`. For preparing GDF data two steps are required:

1. `$ python gdf_data_preparer_normalizer.py`
2. `$ python gdf_data_preparer.py` - this may take a while

To see the some visualizations check `stock_overview.ipynb`


## Queue Imbalance

We use **queue imbalance** to predict **mid price indicator**.
Contains experiments for **queue imbalance** and **previous queue imbalance** features:
    
* `$ python que_log.py` - QUE+LOG (Logistic Regression on **queue imbalance**)
* `$ python que_svm_lin.py` - QUE+SVM with linear kernel
* `$ python que_svm_rbf.py` - QUE+SVM with RBF kernel
* `$ python que_svm_sigmoid.py` - QUE+SVM with sigmoid kernel
* `$ python prev_que_log.py`  - PREV+QUE+LOG (Logistic Regression on **queue imbalance** and **previous queue imbalance**)
* `$ python prev_que_svm_lin.py` - PREV+QUE+SVM on linear kernel
* `$ python prev_que_svm_rbf.py` - PREV+QUE+SVM on RBF kernel
* `$ python prev_que_svm_sigmoid.py` - PREV+QUE+SVM on sigmoid kernel

### Notebooks

* `que_svm.ipynb` - visualizes the results for QUE+SVM
* `prev_que_svm.ipynb` - visualizes the results for PREV+QUE+SVM
* `que_log.ipynb` - visualizes the results for QUE+LOG

## GDF

We use Gaussian Density Filter to extract features from LOB to predict **mid price indicator**. 
We add **queue imbalance** to feature space, then we apply PCA algorithm to reduce dimensionality of the data before 
applying classification algorithm.

### Experiments

For GDF+PCA features we use LSTM, GRU, MLP and Logistic Regression. For Logistic Regression we 
calculate results in a notebook, for the rest of the algorithms the experiments are as follows:

1. Experiments which will calculate one single run of LSTM, GRU and MLP and save results
in `res_gdf_pca_lstm`, `res_gdf_pca_gru` and `res_gdf_pca_mlp` respectively:

* `$ python run_gdf_pca_lstm.py` - calculate single run of GDF+PCA+LSTM
* `$ python run_gdf_pca_gru.py` - calculate single run of GDF+PCA+GRU
* `$ python run_gdf_pca_mlp.py` - calculate single run of GDF+PCA+MLP

2. Experiments to run  LSTM, GRU or MLP classifier with the highest MCC score on 
validation set 30 times are:

* `$ python run_gdf_pca_lstm_iter.py` - calculate best GDF+PCA+LSTM 30 times
* `$ python run_gdf_pca_gru_iter.py` - calculate best GDF+PCA+GRU 30 times
* `$ python run_gdf_pca_mlp_iter.py` - calculate best GDF+PCA+MLP 30 times

Beware that they choose the best classifier based on results in  
`res_gdf_pca_lstm`, `res_gdf_pca_mlp` or `res_gdf_pca_gru` depending on which algorithm 
you have chosen. They save results in `res_gdf_pca_lstm_iter`, `res_gdf_pca_gru_iter` or
`res_gdf_pca_mlp_iter` respectively.

3. To run McNemar Test first you need to generate predictions for LSTM, GRU or MLP:

* `$ python gdf_pca_lstm_predictions.py` - calculates predictions for the best GDF+PCA+LSTM
* `$ python gdf_pca_gru_predictions.py` - calculates predictions for the best GDF+PCA+GRU
* `$ python gdf_pca_mlp_predictions.py`- calculates predictions for the best GDF+PCA+MLP

They will be saved in `res_gdf_pca_lstm_mcnemar`, `res_gdf_pca_gru_mcnemar` or `res_gdf_pca_mlp_mcnemar`
respectively.

Next you can run McNemar Test, which will save results in the same directory in *.csv 
with prefix `mcnemar_':

* `$ python gdf_pca_lstm_mcnemar.py` - calculates McNemar Test for the best GDF+PCA+LSTM and QUE+LOG
* `$ python gdf_pca_gru_mcnemar.py` - calculates McNemar Test for the best GDF+PCA+GRU and QUE+LOG
* `$ python gdf_pca_mlp_mcnemar.py` - calculates McNemar Test for the best GDF+PCA+MLP and QUE+LOG

### Notebooks

To run any notebook make sure you run experiment for QUE+LOG (script `queue_imbalance/que_log.py`),
because it is a baseline algorithm and we compare against it. Also make sure you run the corresponding 
experiment to the notebook you wish to run.

#### LSTM 

* `gdf_pca_lstm.ipynb` - choice of the best GDF+PCA+LSTM
* `gdf_pca_lstm_iter.ipynb` - results after running best GDF+PCA+LSTM 30 times
* `gdf_pca_lstm_mcnemar.ipynb` - McNemar Test for GDF+PCA+LSTM and QUE+LOG

#### GRU

* `gdf_pca_gru.ipynb` - choice of the best GDF+PCA+GRU
* `gdf_pca_gru_iter.ipynb` - results after running best GDF+PCA+GRU 30 times
* `gdf_pca_gru_mcnemar.ipynb` - McNemar Test for GDF+PCA+GRU and QUE+LOG

#### MLP

* `gdf_pca_mlp.ipynb` - choice of the best GDF+PCA+MLP
* `gdf_pca_mlp_iter.ipynb` - results after running best GDF+PCA+MLP 30 times
* `gdf_pca_mlp_mcnemar.ipynb` - McNemar Test for GDF+PCA+MLP and QUE+LOG


## Running experiments

Make sure you have Python 3.6.

Install lob_data_utils:

`$ cd data_utils; python setup.py install'

Install requirements:

`$ pip install -r requirements.txt`

Make sure you have jupyter-notebook installed if you wish to run the notebooks.