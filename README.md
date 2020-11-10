# HyPR-Hybrid-Page-Ranking-on-Evolving-Graphs

In this work, we propose Hybrid Page Rank (HyPR),
which computes PR on evolving graphs using collaborative
executions on muti-core CPUs and massively parallel GPUs. We
exploit data parallelism via efficiently partitioning the graph
into different regions that are affected and unaffected by the
new updates. The different partitions are then processed in an
overlapped manner for PR updates. The novelty of our technique
is in utilizing the hybrid platform to scale the solution to massive
graphs. The technique also provides high performance through
parallel processing of every batch of updates using a parallel
algorithm.

## Building

The project is written in standard C++, OpenMp and CUDA and Python.
To create an evolving graph, the codes are written in Python, and
Pagerank calculation over this evolving graph, the codes are written
in C++ and CUDA.

1) Before PR calcualtion, the dataset(amazon) is divided into x% 
& y% and the y% is divided into z batches. The evolving graph is created for y% of the data and static
Pagerank is calculated for x% of the data using NVgraph.

```
python graph.py dataset x z
```
This creates z batches from the y% dataset.

Arguments:

1. **graph.py**: This is the script for creating batches.
2. **dataset**: This is the static dataset.
3. **x**: The scratch dataset created to calculate PR using NvPagerank.
4. **z**: Number of batches to be created.

2)After the batches are created, run the HyPR algorith to do preprocessing
of the incoming batches with old batch and calculate old, new and border vertex.
This is done parallely using OpenMp on the Host side. As soon as this preproceesing
gets completed, the scaling and Pagerank calculation gets started.

```
nvcc -Xcompiler -fopenmp hyPr.cu -o hyPr -std=c++11 -ljsoncpp
./hyPr 0 ranks_amazon.txt
```

## Getting Started


### Prerequisites

We need to install these libraries for proper functioning of the code

1.  [NetworkX](https://networkx.github.io/) used fo storing graphs
2.  [Numpy](http://www.numpy.org/) used for interpolation of curves
3.  [Json Library] (https://sourceforge.net/projects/jsoncpp/) used as reader and writer in C++.

### Installing

All the above librarie can be easily installed using apt-get in Ubuntu

```
sudo apt-get install python-networkx
sudo apt-get install python-numpy
sudo apt-get install python-matplotlib
```


## Authors

* **Hemant Kumar Giri** - [girihemant19](https://github.com/girihemant19)
* **Mridul Haque**
* **Dip Sankar Banerjee**


## Acknowledgments

* Group Project Partners
