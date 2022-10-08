# LAW2ORDER

## 1. Set up
###1. Set a conda virtual environment
```bash
conda create -n LAW2ORDER python=3.8 anaconda --yes
```

###2. Clone the repository
```bash
git clone https://github.com/ChangYong-Oh/LAW2ORDER.git
```

###3. Install required packages
```bash
conda activate LAW2ORDER
conda install --file requirements.txt
```

####3-1. Structure Learning
```bash
conda install -c r r-base=3.6.1
```
Install R packages
```R
install.packages("bnlearn")
if (!requireNamespace("BiocManager", quietly = TRUE))
    install.packages("BiocManager")
BiocManager::install("graph")
```

###4. Set paths
Set path variables [__DATA_DIR__, __EXP_DIR__] appropriate to your machine in the file below
```bash
LAW2ORDER/experiments/config_file_path.py
```