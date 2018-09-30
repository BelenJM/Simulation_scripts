# -*- coding: cp1252 -*-
import numpy as np
from random import randint, random, choice
from time import gmtime, strftime
import numpy.ma as ma

# Main author: Belen JM
# Contributors: Paula Tataru, Mathieu Tiret

import cProfile
import gc

"""
v8.8: Addition of option for intial allele frequency
v8.7: -Change to store values of NeT and NeHe from different 'interval'
v8.6: Change in generation output: NeLD is stored one generation before compared
 as where it was being stored previously in v8.5.Now it is stored at generation 
 t, whereas before it was stored at t+1 with NeT and NeHe. NeT and NeHe are 
 still stored at generation t+1
v8.5: Elimination of "He_Ne" function and lists He and He2 (in evolution 
 main function) for estimation of the estimators for different intervals
 -Modification in interval_NeHe and interval2_NeHe (change of function to 
 "He_Ne_new")
 -Addition of counter of non-fixed loci at generation t in function "He_Ne_new"
v8.4: "estimation_He" is modified: there is not filtering of the 
 fixed loci
v8.3: new function: Ne_He_new which is a mixture between
 the previous functions of "estimation_He" and "NeHe". In the output,
 the result of this new function is given as: NeHe_new (column 6)
v8.2: addition of temporal samples with time interval 5 and 10,
 modification of output file(addition of oldHe at generation 0 and
 the registration of generations: 
 * at t= -1: pool of individuals with equal alle freq
 * at t= 0: oldpop is selected (start of simulation)
 * at t= 1: newpop is selected



Description:
------------
Script that models the evolution of a diploid bi-locus population
that evolves with genetic drift and whose genes recombine. 
We estimate the effective population size by three different methods:
-Decrease in Heterozygosity (He)-based method: for the computation 
of He we used the method of Nei & Roychoudhury (1974) 
-Temporal method: for the computation Fc (the change in allele frequencies between two different time samples),we used F estimator developed by Nei and Tajima (1981) and Waples (1989).
-Linkage disequilibrium method


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
"""

#-----N, generations, replicates:
N = 5
generations = 9
replicates = 3

#-----Nb of loci, alleles and allele freqs:
# nloc = 50000
nloc = 4
#nall = nb of different alleles (between 1 and 2N)
nall = 2

rr = 0.5
r = np.array([rr]*(nloc-1))
d = 1
interval = 7
#interval2 = 10

fr = 0.5
allele_frequency = np.array((fr,1-fr))

print "Multi-locus model: "
print "N = ",N,",", "generations = ",generations,",","replicates = ",replicates
print "nloc = ", nloc, "nall = ", nall #, 'k1 =', k1


# Printing output file
def outputfile_MAIN(r, N, nloc, nall, replicates,generations, filename):
    global resfile
    ladate = gmtime()
    filename += strftime(".%d-%m-%Y-%H-%M-%S", ladate)
    print filename
    resfile = open(filename, 'w')
    # write header for output file
    resfile.write(70 * "#" + "\n")
    resfile.write("Result file:\t%s\n"% filename)
    resfile.write(strftime("Created:\t%a, %d %b %Y %H:%M:%S +0000\n", ladate))
    # parameters: FILL!
    resfile.write(70 * "#" + "\n")
    # column names
    resfile.write("Rep\tGen\tNb_Alle\tHe\tNb_loci\tNeHe\tFc\tNeT1\tNeT2\tNeT3\tNeHe_02\tNeHe_03\tNeHe_04\tNeHe_05\tNeHe_06\tFc_02\tFc_03\tFc_04\tFc_05\tFc_06\tNeT1_02\tNeT1_03\tNeT1_04\tNeT1_05\tNeT1_06\tr2\tNeLD1\tNeLD2\n")
    return True


filename = 'result_loci%s_nall%s_N%s_recom%s.txt'% (nloc, nall, N, rr)
outputfile_MAIN(r, N, nloc, nall, replicates, generations, filename)

###########################  
def genetic_drift(N, nloc, r, oldpop):
    """
    Simulation of the process of genetic drift, including
    recombination.
    
    Parameters
    ----------
    oldpop = the population at generation t.
    newpop = the population at generation t+1, 
    generated from oldpop.
    parents = random numbers to choose the parents 
    (from which the gametes are chosen)that will generate 
    the newpop 
    rand = random number to simulate the chance of 
    recombination happening between the pair of loci. 
    If the random number <= r, then, there is recombination 
    between that particular pair of loci.
    choice_gamete = random number (0 or 1) to choose which 
    gamete from oldpop at random would be transmitted to 
    newpop.This random number is the index of the gamete that 
    is chosen.
    gametes = the two possible gametes from the selected 
    parent from oldpop
    gamete_res = from the two possible gametes ('gametes'),
    'gamete_res' is the chosen gamete, according to 'choice_gamete'
    gamete_if_recomb = whenever there is a recombination event,
    the "default" gamete remains changed, and the index of the 
    "non-default" gamete is stored here.
    index_array = an array of True/False to form the new gamete 
    after recombination. The first locus of the gamete does not 
    go into recombination, so the array always start with True.
    Note that for every recombination event, we maintain the 
    previous recombination events.
    
    """
    
    newpop = np.empty(oldpop.shape, dtype=np.int)
    
    parents = np.random.randint(0,N,(N,2))

    # Genetic drift & recombination
    for person in range(N): #we form the newpop by individuals...
        for allele in range(2): #...and alleles
            rand = np.random.random_sample((nloc-1)) <= r 
            # Array of True or False if there is or not recombination event
            choice_gamete = np.random.randint(0,2)
            # Choice of gamete: either 0 or 1

            gametes = oldpop[parents[person,allele], :, :] 
            # parents[person,parent] == the parent chosen by random
            
            gamete_res = gametes[:,choice_gamete] 
            # From the two possible gametes, the "default" gamete is chosen
              
            gamete_if_recomb = 1-choice_gamete
            # In case there is recombination, the "default" gamete is changed
            
            # Recombination:
            for recombination_event in range(nloc-1):
                # update indexes by switching when
                # the random number is smaller than the
                # recombination probability     
                if rand[recombination_event] == True: 
                # if it is True, there is recombination
                    index_array = np.append(np.array([True]*(recombination_event+1)), np.array([False]*(nloc-recombination_event-1)))

                    gamete_res = np.where(index_array, gamete_res, gametes[:,gamete_if_recomb])
                    #wherever in "index_array" there is a True,
                    #puts "gamete_res"; if there is a False, gametes

                    gamete_if_recomb = 1-gamete_if_recomb 
                    # Now, next time there is recombination, 
                    # this is the index of the other gamete 
                           
            newpop[person,:,allele] = gamete_res

    return newpop

def count_alleleFrequency(indiv):      
    """
    It counts the allele frequency of indiv.

    Parameters
    ----------
    allelefreq = allele frequencies of nloc loci
    nloc = number of loci
    nall = number of alleles in each locus
    N2 = number of total alleles in the pop at each generation(2*N)

    """
    # np.empty is faster than np.zeros
    allelefreq = np.empty((nloc, nall))
    N2 = 2.0*indiv.shape[0]

    for loc in range(nloc):
        # instead of reshape, using flatten, might be faster
        allelefreq[loc] = np.bincount(indiv[:, loc].flatten(), minlength=nall)/N2

    return allelefreq
    
def estimation_He(p):
    """
    Estimation of Heterozygosity 

    Parameters
    ----------
    p = allele frequencies for each locus for all loci
    Het = expected heterozygosity averaged over loci, according to Nei & Roychoudhury (1974).

    Method (Nei & Roychoudhury, 1974):
    --------
    -Average heterozygosity per locus (H):
    The mean of the heterozygosity in each locus, over all
    structural loci in the genome.    

    * Based on paper:
    Nei, M. & Roychoudhury, A.K (1974)
    Sampling variances of heterozygosity and genetic distance.
    Genetics, 76: 379 - 390.
    
    Notes:
    -------
    --> Case of nall=2 : [0,1]: average(1 - sum(p**2))= 0.5
    --> Case of nall=2*N : (range(0,2*N)): average(1 - sum(p**2)) = 1
    """
    #print "p", p
    #print type(p)
    # test to see if the sum of allele freq per loci=1:
    #print "p(locus)", np.sum(p, axis=1)
  
    # Average He over loci
    Het = np.average(1 - (np.sum(p**2, axis=1)))
    #print "He:", Het
    return Het
    
    
def He_Ne_new(allelefreq_oldpop, allelefreq_newpop, d):
    """
    This function:
    1.Filtering for fixed alleles at generation t. 
    Loci with alleles with allele frequency ==1 at generation t 
    will be removed (from t and t+1).
    2.Counts the nb of non-fixed loci at generation t
    3.Estimation of He at t and t+d of the remaining loci
    4.estimates rate of decrease in He (DeltaH):
        deltaH = (oldHe - newHe) / oldHe
        deltaH = (1-(newHe/oldHe))/d
    5.estimates Ne from DeltaH (NeHe):
        NeHe = .5 / deltaH
    """
    #print "pre-filtering all freq oldpop:", allelefreq_oldpop
    # OLDPOP:
    # Filtering for fixed and missing alleles at generation t (oldpop):
    filtered_allelefreq_oldpop = [allelefreq_oldpop[locus,:] for locus in range(len(allelefreq_oldpop)) if (allelefreq_oldpop[locus,:]==1).sum() == 0 ]
    #print "post-filtering all freq oldpop:", filtered_allelefreq_oldpop

    # Reshaping it after filtering
    filtered_allelefreq_oldpop = np.reshape(filtered_allelefreq_oldpop, (len(filtered_allelefreq_oldpop), nall))

    # NEWPOP:
    # Filtering allelefreq_newpop based on oldpop:
    #print "pre-filtering all freq newpop:", allelefreq_newpop
    changed_allelefreq_newpop = [allelefreq_newpop[locus,:] for locus in range(len(allelefreq_oldpop)) if (allelefreq_oldpop[locus,:]==1).sum() == 0 ]
    #print "post-filtering all freq newpop:", changed_allelefreq_newpop
    
    # Reshaping it after filtering
    changed_allelefreq_newpop = np.reshape(changed_allelefreq_newpop, (len(changed_allelefreq_newpop), nall))
    #print "Reshaped filtered newpop:", changed_allelefreq_newpop
    
    # Count the number of loci non-fixed at generation t:
    number_non_fixed_loci = len(filtered_allelefreq_oldpop)
    
    # He:
    # Average He over loci
    oldHe = np.average(1 - (np.sum(filtered_allelefreq_oldpop**2, axis=1)))
    newHe = np.average(1 - (np.sum(changed_allelefreq_newpop**2, axis=1)))
    #print 'oldHe v8.3 = ', oldHe
    #print 'newHe v8.3 = ', newHe
    
    # DeltaHe:
    deltaH = (1 - (newHe/oldHe)) / d

    # NeHe:
    if deltaH != 0:
        NeHe = .5 / deltaH
    else:
        NeHe = float('nan')
    
    #print "new deltaH:", deltaH
    #print "new NeHe:", NeHe
    return number_non_fixed_loci, NeHe   
    
    
def r_square(pA, pB, pAB):
        """"
        Computation of r squared per pair of loci.
    
        Parameters
        ----------
        pAB = frequency of gametes AB (00)
        pA = frequency of allele A (allele type 0)
            *Nb of alleA/total genes per locus
            (always 2*N in this case)
        pB = frequency of allele B (allele type 1)
            *Nb of alleB/total genes per locus
            (always 2*N in this case)
        
        """

        D = (pAB - (pA * pB))**2
        den = (pA*(1-pA) * pB*(1-pB))
        if den == 0: # to avoid division by 0
            return 0
        return D/den

def LD(indiv, nloc):   
    """
    Estimation of LD between the loci of indiv.
    
    Parameters
    ----------
    comb = number of pair of loci for which LD will be computed
    N2 = number of total alleles in the pop at each generation(2*N)
    flat_locus = flatten array of the alleles of each locus
    r_sq = r squared of all the combinations of pair of loci
     for which we compute LD
    locus_i = one of the loci from the pair of loci for 
    which we compute LD
    locus_j = the other locus from the pair of loci for 
    which we compute LD
    count_00 = number of gametes type 00
    count_01 = number of gametes type 01
    count_10 = number of gametes type 10
    pAB = frequency of gametes AB (00)
    pA = frequency of allele A (allele type 0)
        *Nb of alleA/total genes per locus(always 2*N in this case)
    pB = frequency of allele B (allele type 1)
        *Nb of alleB/total genes per locus(always 2*N in this case)
    this_r = LD (r squared) between this pair of loci 
    (locus_i and locus_j).
    rsquare_allloci = LD (r squared) averaged over all 
    combination of pair of loci for which we estimate LD
    Ne_LD = Ne from LD, according to first of equations 7 of 
    Weir & Hill(1980). In the paper, N is sample size: we are
     sampling all individuals, so, sample size=N.
    Ne_LD2 = Ne from LD, according to second of equations 7 of 
    Weir & Hill(1980)
    """
    comb = 0
    N2 = 2.0*indiv.shape[0]

    #Flatting the array for each locus:
    flat_locus = [indiv[:,i,:].flatten() for i in range(nloc)]

    #For each combination...
    r_sq = 0
    for i in range(nloc-1):
        locus_i = flat_locus[i]
        for j in range(i+1, nloc):
            locus_j = flat_locus[j]
            
            #Counting gametes:
            count_00 = np.count_nonzero((locus_i==0) & (locus_j==0))
            count_01 = np.count_nonzero((locus_i==0) & (locus_j==1))
            count_10 = np.count_nonzero((locus_i==1) & (locus_j==0))
            #Frequency of the gametes:
            pA = (count_00 + count_01)/N2
            pB = (count_00 + count_10)/N2
            pAB = (count_00)/N2

            # Whenever a loci is fixed, (pA*(1-pA)*pB*(1-pB)) == 0,
            # LD will be 0
            # Only computing LD between loci not fixed:
            if (pA*(1-pA)*pB*(1-pB)) != 0:
                this_r = r_square(pA, pB, pAB)
                comb += 1
                r_sq += this_r
            #if this_r < 1e-8:
                # PAULA I reached a point where LD(i,j) is almost 0
                # we know that LD(i,k)<=LD(i,j) with k > j
                # so break the inner for loop, the one over j
                # break
                # BELEN With this_r == 1e-8, this condition works ok
                # for nloc = 3,10,20  
    
    if comb == 0: # in case all pair of loci are fixed
        #print "comb ==0", comb 
        rsquare_allloci = 0
        Ne_LD = 0
        Ne_LD2 = 0
    
    else:
        rsquare_allloci = r_sq/comb # averaging over pair of loci
    
    #Computation of Ne:
        Ne_LD = 1.0/(3.0*(rsquare_allloci-(0.5/N)))

    #Computation of Ne_LD2:
        Ne_LD2 = 1.0/(3.0*(rsquare_allloci-(1.0/float(N))))

    return rsquare_allloci, Ne_LD, Ne_LD2

    
def Fc_Ne(allelefreq_oldpop, allelefreq_newpop, d):
    """"
    This function computes the change in allele frequency
    of the population between 2 generations, according to
    Waples(1989), separated a time interval d.
    
    1.Filtering for fixed alleles at generation t. 
    Alleles with allele frequency == 0 or 1 at generation t 
    will be removed.
    2.Estimation of Fc:
     2.1 Fc is estimated by locus:
        xi = allele freq at oldpop(generation t)
        yi = allele freq at newpop(generation t+d)
     2.2 and averaged per all loci (in this script, we average
     over all alleles of all loci because we estimate a Fc per 
     allele).
    3.Estimation of Ne: according to
    
    Method (Nei & Tajima, 1981) and (Waples, 1989):
    ------
    Allele frequency change (Fc) computed averaged over all alleles,
    weighted by all loci. 

    Reference paper:
    ----------------
    Waples, R. (1989) A generalized approach for estimating
    effective population size from temporal changes in allele
    frequency. Genetics, 121(2): 379-391.
    
    Parameters
    ----------
        count_nb_alleles = number of alleles
        xi_old = allele frequencies of the locus at time t.
        yi_new = allele frequencies of the locus at time t+d.
        xi = filtered allele frequencies of the locus at time t.
        yi = filtered allele frequencies of the locus at time t+d.
        Fc = frequency change of all the alleles of the locus.
        FcBar_average = average of FcBar weighted by all alleles of
        all loci who were not fixed at generation t
        (same as doing FcBar weighted first by alleles per locus
        and then, over all loci). 
        d = time interval between the 2 samples
        Ne1, Ne2, Ne3 = different estimators of Ne according 
        to Waples (1989); in previous versions of script, these
        are called Ne2, Ne3 and Ne4
    
    Reminder:
    *Ne resulted is the result from:
    AF of gener 0 and 1--> give Fc at gener 0--> gives Ne at gener 0
    """""
    
    # empty is faster than zeros
    FcBar = np.empty(nloc)
    # range is by default from 0
    count_nb_alleles = 0
    Fc_stored = 0.0
    for loc in range(nloc):
        yi_new = allelefreq_newpop[loc]
        xi_old = allelefreq_oldpop[loc]
        
        # filtering alleles:
        yi = yi_new[np.nonzero((xi_old > 0) & (xi_old < 1))]
        xi = xi_old[np.nonzero((xi_old > 0) & (xi_old < 1))]
        
        # PAULA this here gives error sometimes
        # says invalid value encountered in divide
        # and I find a 0 in (.5 * (xi + yi) - (xi * yi))
        # this happens when xi = yi = 1
        # you need a check
        
        #den = (.5 * (xi + yi) - (xi * yi))
        #print "den=", den
        #pos = np.nonzero(den) # Check!!!!!
        #Fc = (xi[pos] - yi[pos])**2 / den[pos]

        den = (.5 * (xi + yi) - (xi * yi))
        #print "den=", den
        
        if len(den) != 0:
            Fc = (xi - yi)**2 / den
            Fc_stored += np.sum(Fc)
            count_nb_alleles += np.count_nonzero(xi)
            #counting number of segregating alleles
            
    #average over alleles of all loci:
    if count_nb_alleles != 0:
        FcBar_average = Fc_stored/count_nb_alleles   
    
        if FcBar_average != 0:
        #plan I, eq 12 (Waples 1989):
            Ne1 = (.5 * d) / FcBar_average
            #print "Ne1_new =", Ne1
        else:
            Ne1 = float('nan')
        
        if (FcBar_average - 1./N) != 0:
            #plan II, eq 11 (Waples 1989)(=Fred's script Ne2):
            Ne2 = (.5 * d) / (FcBar_average - 1. / N)
            #print "Ne2 =", Ne2
            #plan I, eq 13 (Waples 1989):
            Ne3 = (.5 * (d-2))/(FcBar_average - 1. / N)
            #print "Ne3 =", Ne3
        else:
            Ne2 = float('nan')
            Ne3 = float('nan')
    else:
        FcBar_average = float('nan')
        Ne1 = float('nan')
        Ne2 = float('nan')
        Ne3 = float('nan')
    #print FcBar_average, Ne1, Ne2, Ne3
    return FcBar_average, Ne1, Ne2, Ne3
    

def evolution(replicates):
    """
    Function that makes the population evolve at each
    generation.

    Parameters
    ----------
    rep = counter of replicates
    oldpop = array of arrays with shape (N, nloc, 2): individuals,
    number of loci, two alleles
    allelefreq_oldpop = allele frequencies per loci of oldpop. 
    He_old = Heterozygosity of oldpop
    counter_generations = counter of generations
    counter_alleles = number of alleles of oldpop
    Ne_LD = Ne estimated from LD function (LD method: NeLD 
    and NeLD2)
    newpop = array of arrays with shape(N, nloc, 2), resulted
    from oldpop after passing through genetic drift process
    allelefreq_newpop = allele frequencies per loci of newpop
    He_new = Heterozygosity of newpop
    NeHe = Ne estimated from He_Ne function 
    (rate of decrease in heterozygosity-based method)
    NeFc = Ne estimated from Fc_Ne function (temporal method)

    """
    
    #For each replicate...:
    for rep in range(replicates):
        #print "replic:", rep
        # PAULA
        # oldpop = np.random.randint(nall, size=(N*nloc,2))
        # oldpop = oldpop.reshape(N, nloc, 2)    
        #oldpop = np.random.randint(0, nall, (N,nloc,2))
        oldpop = np.random.choice(nall,(N,nloc,2),p=allele_frequency)
        
        allelefreq_oldpop = count_alleleFrequency(oldpop)
        #print "allele freq oldpop:", allelefreq_oldpop
        He_old = estimation_He(allelefreq_oldpop)
        # Print He at generation 0 after creating the population:
        if rep != 0:
            resfile.write("\n")
        resfile.write("%2i\t%2i\t%2i\t%7f\t" % (rep,0,len(np.unique(oldpop)), He_old))
        resfile.write("nan\t"*(6+((interval-2)*3)))
        #print "He old:", He_old
        #resfile.write("%7f\t" % (He_old))
        
        # store all the simulated frequencies in a list,
        # together with the corresponding He
        # for starters, they only contain the values for oldpop
        allelefreq = [allelefreq_oldpop]
        #print "allele freq 2:", allelefreq
        He = [estimation_He(allelefreq_oldpop)]

        # ...in each generation:
        # a for loop should be better than while
        # counter_generations = 0
        # while counter_generations < generations:
        
        for counter_generations in range(generations):
            # storing the estimators inside generations
            # with interval==5
            interval_NeHe = 0
            interval_NeFc = 0
            #count nb of alleles from generation 0:
            counter_alleles = len(np.unique(oldpop))

            #Computation of LD(r**2) and Ne
            Ne_LD = LD(oldpop, nloc)
            #print "NeLD:", Ne_LD
            # writing NeLD in generation t, before genetic drift
            resfile.write("%5f\t%5f\t%5f" % (Ne_LD))
            resfile.write("\n")
            
            #Genetic Drift:
            newpop = genetic_drift(N, nloc, r, oldpop)
            # delete oldpop to save some space
            del oldpop
            # call garbage collector
            gc.collect()

            #counting allele freq of newpop after genetic drift:
            allelefreq_newpop = count_alleleFrequency(newpop)
            
            #Estimation of He:
            He_new = estimation_He(allelefreq_newpop)  
            
            #Computation of HeNe:
            #NeHe = He_Ne(He_old, He_new, d)
            #print "NeHe:", NeHe
            
            #Estimation of newNeHe:
            NeHe_new = He_Ne_new(allelefreq_oldpop, allelefreq_newpop, d)
                        
            #Computation of TempoNe:
            # this function gives some warnings
            NeFc = Fc_Ne(allelefreq_oldpop, allelefreq_newpop, d)
                  
            # WRITING PART:
            resfile.write("%2i\t%2i\t%2i\t" % (rep,counter_generations+1,counter_alleles))
            resfile.write("%7f\t" % (He_new))
            resfile.write("%2i\t%7f\t" % (NeHe_new))
            #resfile.write("%5f\t%5f\t%5f\t" % (Ne_LD))
            resfile.write("%7f\t%7f\t%7f\t%7f\t" % (NeFc))
            
            # calculate NeHe and NeFc for an interval>2
            if len(allelefreq) == interval:
                for year in range(2, interval):   
                   #temporal_NeHe.append(He_Ne(He_new[0], He_new[year], year))

                   interval_NeHe = He_Ne_new(allelefreq[0], allelefreq[year], year)
                   #print "NeHe after 5 generations:", interval_NeHe
                   interval_NeFc = Fc_Ne(allelefreq[0], allelefreq[year], year)
                   #print "TempoNe after 5 generations:", interval_NeFc
                   # Writing:
                   resfile.write("%7f\t" % (interval_NeHe[1]))
                   resfile.write("%7f\t%7f\t" % (interval_NeFc[0],interval_NeFc[1]))

            else:
                resfile.write("nan\t"*(interval-2))
                resfile.write("nan\t nan\t" * (interval-2))


            # I need to append at the end the values from newpop
            allelefreq.append(allelefreq_newpop)
            #print "updating the matrix with a new allelefreq:", allelefreq

         
            # finish the last line of output that would correspond 
            # to NeLD at the last generation of the replicate:
            if counter_generations == (generations-1):
                Ne_LD = LD(newpop, nloc)
                resfile.write("%5f\t%5f\t%5f" % (Ne_LD))
                
            oldpop = newpop
            allelefreq_oldpop = allelefreq_newpop
            He_old = He_new
            

#Here we run the main program: Evol
evolution(replicates)

# cProfile measures the running time in each function
# helpful to find out where most of the time is spent
# cProfile.run('evolution(replicates)', sort='time')

#stop = time.time()
#print 'Performance of "Multi-loc model script":', stop-start

resfile.close()

