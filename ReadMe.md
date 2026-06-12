# CASANDRA: Chemical Protein Association Score Predictor Using AI Agent Skills

Requirements:
1. python
2. biopython
3. Rdkit
4. Protllm
5. ChemBERT-2a



## Data Statistics:
1. Number of unique species dataset: 2030
2. Number of unique chemicals in raw data before trimming for molecular weight: 64690525
3. Number of unique chemicals after trimming: 61456131 (95%)
### 1. Understanding chemicals in the dataset

Moelcular weight quantiles 
df["molecular_weight"].quantile(0.025)
181.14548
>>> df["molecular_weight"].quantile(0.975)
745.9257

(5% total removed from both ends)
