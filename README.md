# PETIL: Predicting Expansion of Tumor Infiltrating Lymphocytes for the Adoptive Cell Immunotherapy in Bladder Cancers

One major advance in treating solid tumors is the success of adoptive cell therapy (ACT), in which autologous tumor-infiltrating lymphocytes (TILs) are expanded and activated ex vivo and then reinfused into the cancer patient. 

PETIL is a tool that can first learn from patient and tumor data already collected in the clinic (local data) which data features are important for predicting TIL expansion, without the need to predefine which data categories to consider. Then, this tool predicts a possible TIL expansion for individual patients (personalized predictions), allowing to determine whether ACTTIL therapy could potentially treat an individual bladder cancer patient.


## PETIL needs the following libraries

```bash
numpy
sklearn
matplotlib
seaborn
pandas
tensorflow
statsmodels
scipy
```


## Implementing PETIL

PETIL is implemented in the following order:

01_Data_summary.ipynb  
01b_Data_summary_cli_Trl.ipynb  
02_Data_table.ipynb  
02b_Data_table_cli_Trl.ipynb  
03_norm.ipynb  
03b_norm_cli_Trl.ipynb  
04_midas_train.ipynb  
04b_midas_test.ipynb  
05_MI.ipynb  
06_p_corr.ipynb  
07_adeq_smplSz.ipynb  
08_FFS.ipynb  
09_MCC_RBF_SVM.ipynb


## Authors

Kayode Olumoyin kayode.olumoyin@moffitt.org, Katarzyna Rejniak 


## Source Code
https://github.com/okayode/Predictor_of_the_Expansion_of_TIL_project

## License


This project is licensed under the GNU General Public License v3.0.
