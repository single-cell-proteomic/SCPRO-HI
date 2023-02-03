# Integration of Single-cell Proteomic Datasets Through Distinctive Proteins in Cell Clusters
This repository contains the code of SCPRO-HI algorithm and more for reproduction of experimental results in the submitted paper.

## System requirements

 The code is tested on Python version >= 3.6. Following Python packages are required:
- Pip3
- Pandas
- Numpy
- Scikit-learn
- NetworkX
- matplotlib
- TensorFlow / Keras
- scanpy 
- scipy
- scib
- umap


The packages shown below are installed automatically by the code using Pip3:
- harmonypy
- mnnpy
- scanorama
- scvi-tools
- pyMARIO

## Usage

**Data format**

The SCPRO-HI algorithm and reproduction pipeline accepts single-cell datasets in annData format (with .h5ad file extension). All the datasets will be integrated have to be in the same directory and the path of that directory is given to the driver function.

**Demo**

Run the **SCPRO.py** file in the command line by passing **SCPRO-HI** as method parameter and the path of the single-cell datasets' directory, respectively. Please note that, if **All** is provided for the 'method' parameter, than all the algorithms in the paper are executed.

**Quick start**

The model can be tested by running **SCPRO.py** file in command line or calling **scprohi_run()** function from **SCPRO.py** file in the code block. Use the following command for calling in Terminal:

`$ python3 SCPRO.py "SCPRO-HI" datasets_directory_path

or just calling as a function:

`scData = scprohi_run(method, data_path)`

In both cases, the algorithm returns a scData object which is a wrapper for holding both individual annData objects of each datasets and the concatenated annData object of them.
The concatenated dataset is stored in **.whole** feature of scData object and the list of the individual datasets is accesiable in **.dataset_list** feature.
The integrated measurements are stored in **.obsm** of the concatenated dataset with the name of the integration method as the key.

Please see the **tutorial.ipynb** notebook for an example usage of the proposed algorithm and also how to reproduce the results in the paper.

