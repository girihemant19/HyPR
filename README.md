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
Pagerank is calculated for x% of the data using [nvGraph](https://github.com/rapidsai/nvgraph).

```
python graph.py dataset x z
```
This creates z batches from the y% dataset.

Arguments:

1. **graph.py**: This is the script for creating batches.
2. **dataset**: This is the static dataset.
3. **x**: The scratch dataset created to calculate PR using [nvGraph](https://github.com/rapidsai/nvgraph).
4. **z**: Number of batches to be created.

2)After the batches are created, run the HyPR algorith to do preprocessing
of the incoming batches with old batch and calculate old, new and border vertexes.
This is done parallely using OpenMp on the Host side. As soon as this preproceesing
gets completed, the scaling and Pagerank calculation gets started using HyPr Algorithm.

```
nvcc -Xcompiler -fopenmp hyPr.cu -o hyPr -std=c++11 -ljsoncpp
./hyPr 0 ranks.txt
```

## Getting Started


### Prerequisites

We need to install these libraries for proper functioning of the code

1.  [Json Library](https://sourceforge.net/projects/jsoncpp/) used as reader and writer in C++.
2.  [NetworkX](https://networkx.github.io/) used fo storing graphs
3.  [Numpy](http://www.numpy.org/) used for interpolation of curves

### Installing

All the above library can be easily installed using apt-get for python and for JsonLibray download directly from the link given.

```
[Json Library](https://sourceforge.net/projects/jsoncpp/) -->Download from here
sudo apt-get install python-networkx
sudo apt-get install python-numpy
```
Compiler requirement:

* `gcc`     version 4.8+
* `nvcc`    version 9.2


CUDA requirement:

* CUDA 10.2+
* NVIDIA driver 410.48+
* Volta architecture or better

You can obtain CUDA from [https://developer.nvidia.com/cuda-downloads](https://developer.nvidia.com/cuda-downloads).
Compiler requirements:


## Cite this work as

```html
@inproceedings{ giri2020,
  title={ Cite this work as },
  author={Giri, Hemant Kumar and Haque, Mridul and Banerjee, Dip Sankar },
  booktitle={ IEEE 27th International Conference on High Performance Computing (HiPC)},
  year={2020},
  organization={IEEE}
}
```

