## Simulation_scripts
This folder contains the simulation programs written in Python to model the evolution of ideal populations under a set of scenarios, and measure the effective population size using different estimators.

# EvolNe.py 

Description:
------------
Script that models the evolution of a diploid bi-locus population
that evolves with genetic drift and whose genes recombine. 
We estimate the effective population size by three different methods:
(1) Decrease in Heterozygosity (He)-based method: for the computation 
of He we used the method of Nei & Roychoudhury (1974);  
(2) Temporal method: for the computation Fc (the change in allele frequencies between two different time samples),we used F estimator developed by Nei and Tajima (1981) and Waples (1989);
(3) Linkage disequilibrium method

Parameters
----------
N = number of individuals of the population
generations = number of generations the population is going to evolve
replicates = number of replicates of the process
nloc = number of loci
nall = number of different alleles in each locus
totall = number of total alleles in the pop in each generation
rr = recombination rate between each pair of next-to loci
d = time interval between two samples (for temporal method and NeHe). 
In general,d=1
interval = time interval between two samples (for temporal method and NeHe). 
This can be modified to study the performance of the FcNe and HeNe when time 
interval is increased.
fr = initial allele frequency of allele 0

Functionalities of NumpyArrays are used.
