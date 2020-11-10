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