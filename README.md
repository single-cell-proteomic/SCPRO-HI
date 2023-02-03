# Integration of Single-cell Proteomic Datasets Through Distinctive Proteins in Cell Clusters
This repository contains the code for the SCPRO-HI algorithm and additional materials to reproduce the experimental results described in the submitted paper.

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


The following packages are automatically installed by the code using Pip3:

- harmonypy
- mnnpy
- scanorama
- scvi-tools
- pyMARIO

## Usage

**Data format**

The SCPRO-HI algorithm and reproduction pipeline accept single-cell datasets in annData format (with the .h5ad file extension). All datasets to be integrated must be located in the same directory and the path to that directory must be provided to the driver function.

**Demo**

To run the **SCPRO.py** file in the command line, pass 'SCPRO-HI' as the method parameter and the path to the directory containing the single-cell datasets. If you would like to execute all the algorithms in the paper, provide 'All' as the value for the 'method' parameter. Note that this will run all algorithms.

**Quick start**

The algorithm can be tested by either running the **SCPRO.py** file in the command line or calling the **scprohi_run()** function from the **SCPRO.py** file in a code block. To run the file in the Terminal, use the following command:

`$ python3 SCPRO.py "SCPRO-HI" datasets_directory_path

or just calling as a function:

`scData = scprohi_run(method, data_path)`

In both cases, the algorithm returns a scData object, which serves as a container for the individual annData objects of each dataset and the concatenated annData object of all datasets. The concatenated dataset is stored in the .whole attribute of the scData object, and the list of individual datasets can be accessed through the .dataset_list attribute.

The integrated measurements are stored in the **.obsm** attribute of the concatenated dataset, with the name of the integration method serving as the key. For an example of how to use the proposed algorithm and reproduce the results in the paper, please refer to the **tutorial.ipynb** notebook.

