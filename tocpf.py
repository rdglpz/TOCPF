#!/usr/bin/env python3

#Object Oriented implementation for computing the
#TOC= Total Operating Characteristic Curve, functions for probabilistic analysis
#Author: S. Ivvan Valdez
#Centro de Investigación en Ciencias de Información Geoespacial AC
#Querétaro, México.
import numpy as np
import tensorflow as tf
import math
import scipy.linalg as la
import copy
import gc
import rasterio
from rasterio.transform import from_origin

#import annfit as af
#from rasterio import CRS


#Object to store a TOC curve
class TOCPF:
    """
    This class implements the Total Operating Characteristic Curve computation and analysis using probability functions. That is to say, a cumulative distribution is derived from the TOC, then, a mass or density probability function is computed from it.

    :param rank: The class is instantiated with the optional parameter ``rank`` that is a numpy array of a predicting feature.

    :param groundtruth: The class is instantiated with the optional parameter ``groundtruth`` that is a numpy array of binary labels (0,1).

    :cvar kind: A string with the value 'None' by default, it indicates the kind of TOC, for instance: continuous, discrete, semicontinuous, or None for an empty TOC. A semicontinuous TOC is that with continuous disjoint segments.

    :cvar area: The area under the curve of the TOC, in discrete cases it is the sum of heights. Notice that this area is not normalized neither is the rate of the parallelepiped.

    :cvar areaRatio: The areaRation with respect to the parallelepiped area.

    :cvar isorted: All the variables of indices start with an "i", they could be either a true-false array (true in the selected indices), or an integer array. In this case this array stores the indices of the sorted rank, they are sorted from the minimum to the maximum.

    :cvar ndata: All the sizing variables start with "n", in this case, ndata is the total number of data, that is to say the "rank" and "groundtruth" sizes

    :return: The class instance, if ``rank`` and ``groundtruth`` are given it computes the TOC, otherwise it is an empty class.

    :rtype: ``TOCPF``

    """


    area=0
    """
    _`area`=     under the curve of the TOC substracting the right triangle of the paralellogram.

    """
    Darea=0
    """
    _`Darea`, since te TOC is approximated via a dicrete curve, this the approximation of the `area`_ (area under the TOC minus the right parellelogram triangle area), it is useful to determine whether the approximation is sufficiently close to the actual value.

    """

    areaRatio=0
    """
    _`areaRatio` with respect to the parallelepiped. The maximum TOC area is that of the parallelepiped, hence this is the ratio of the TOC area inside the parallelepiped divided by the parallelepiped area, hence its value is between 0 and 1.
    Notice that the area inside the parallelepiped is usually different from the area under the curve, it is computed substracting from the ``area`` variable the triangle in the left side of the parallelepiped.

    """
    areaDRatio=0
    """
    _`areaRatio` with respect to the parallelepiped. The maximum TOC area is that of the parallelepiped, hence this is the ratio of the TOC area inside the parallelepiped divided by the parallelepiped area, hence its value is between 0 and 1.
    Notice that the area inside the parallelepiped is usually different from the area under the curve, it is computed substracting from the ``area`` variable the triangle in the left side of the parallelepiped_`areaRatio` with respect to the parallelepiped. The maximum TOC area is that of the parallelepiped, hence this is the ratio of the TOC area inside the parallelepiped divided by the parallelepiped area, hence its value is between 0 and 1.
    Notice that the area inside the parallelepiped is usually different from the area under the curve, it is computed substracting from the ``area`` variable the triangle in the left side of the parallelepiped

    """


    kind='None'
    """
        The _`kind` attribute indicates the type of TOC curve, that is to say: "continuous","discrete","semicontinuous","forcedContinuous". The continuous is a continuous approximation of the Hits and Hits plus False Alarms, while the discrete is computed when discrete rank values are detected, hence a kind of cummulative histogram is computed, and the cumulative probability function is a cummulative histogram of frequencies, and the probability function is discrete as well. In the "semicontinuous case, it could have segments of contiinuous domains and discrete points, most of the time if ``forceContinuity``_ is ``True``, a semincotinuous TOC is converted to "forcedContinuous", that is to say the discrete approximation is continuous, but it is a repaired semicontinuous curve.
    """

    isorted=None
    """
       The _`isorted` attibute stores the indices fo the sorted rank. Storing the indices is useful to save computational time when converting a rank array to a probability value.

    """

    ndata=0
    """
        _`ndata` stores the number of data in the arrays of the class,  notice that the number of positives (``np``) or other counts are altered when interpolations, normalization or vectorization of the TOC is applied, while ``ndata`` stores the length of the data arrays independent of the mentioned counts.

    """

    np=0
    """
    _`np` stores the number of positive data (1-labeled data, or data with presence of a characteristic).

    """

    PDataProp=0
    """
    _`PDataProp` is the proportion of positive (1-labeled)data in the data. The purpose is to maintain the proportion of class 1 data in TOC operations, hence to preserve the knowlkedege about data imbalance and proportion of classes even if the TOC is normalized.

    """


    HpFA=None
    """
    _`HpFA` is a numpy array with the sum of Hits plus False Alarms.

    """

    Hits=None
    """
    _`Hits` is a numpy array with the sum of true positives

    """

    Thresholds=None
    """
    _`Thresholds` is a numpy array with thresholds of the TOC, they are computed using the ranks, that is to say, most of the times they are equal to the ranks.

    """

    ndiscretization=0
    """
    _`ndiscretization` is a number of segments lower or equal to the number of data, it is used to compute a TOC approximation for a better visualization fo the TOC changes with respect to the rank. If continuous the discrete TOC is used for deriving a density probability function.
    """

    dHits=None
    """
    _`dHits` in a continuous TOC it stores an array of size `ndiscretization`_ that represents the TOC curve. Notice that, often, the TOC curve coordinates are computed at spacing of 1 in the Hits plus False Alarms axis, and 0 or 1 in the Hits axis, this leads to noisy estimations of the probability functions, hence the discretization stored in `dHits`_ and `dHpFA`_ is intended for less noisy but precise representations that are suitable for fine-grain meaningful computations, nevertheless, these represantations could be noisy for a visual analysis of the probability functions, hence, for visual analysis it is preferred to use the `smoothHits`_ and `smoothHpFA`_ representation.
    In a discrete TOC this array is the same that Hits and stores all the discrete coordinates of the TOC.
    """

    dHpFA=None
    """
    _`dHpFA` in a continuous TOC it stores an array of size `ndiscretization`_ that represents the TOC curve. Notice that, often, the TOC curve coordinates are computed at spacing of 1 in the Hits plus False Alarms axis, and 0 or 1 in the Hits axis, this leads to noisy estimations of the probability functions, hence the discretization stored in `dHits`_ and `dHpFA`_ is intended for less noisy but precise representations that are suitable for fine-grain meaningful computations, nevertheless, these represantations could be noisy for a visual analysis of the probability functions, hence, for visual analysis it is preferred to use the `smoothHits`_ and `smoothHpFA`_ representation.
    In a discrete TOC this array is the same that HpFA and stores all the discrete coordinates of the TOC.
    """

    drank=None
    """
    _`drank` in a continuous TOC it stores an array of size `ndiscretization`_ that represents the TOC curve in the rank/threshold domain. Notice that, often, the TOC curve coordinates are computed at spacing of 1 in the Hits plus False Alarms axis, and 0 or 1 in the Hits axis, this leads to noisy estimations of the probability functions, hence the discretization stored in `dHits`_ , `dHpFA`_ `drank`_ is intended for less noisy but precise representations that are suitable for fine-grain meaningful computations, nevertheless, these represantations could be noisy for a visual analysis of the probability functions, hence, for visual analysis it is preferred to use the `smoothHits`_ and `smoothHpFA`_ representation.
    In a discrete TOC this array is the same that HpFA and stores all the discrete coordinates of the TOC.
    """

    dndata=None
    """
    _`dndata` stores the number of data used in the TOC discretization by each discrete point. It is of special interest for discrete TOC, becaue it represents the frequency of each discrete bin.
    """

    CDF=None
    """
    _`CPF` is the conditional cumulative distribution function. CPF=PF(x<=Threshold | y=1). It is of `ndiscretization`_ size, and it is a (rank,Hits) function, that is to say the domain is Hits, and the image is on rank probability.
    """

    PF=None
    """
    _`PF` is the conditional density probability function. PF=p(x<=Threshold | y=1). It is of `ndiscretization`_ size, and it is a (rank,Hits) function, that is to say the domain is Hits, and the image is on rank probability.
    """
    DPF=None
    """
    _`DPF` is the firsat derivative of the conditional density probability function. PF=p(x<=Threshold | y=1). It is of `ndiscretization`_ size, and it is a (rank,Hits) function, that is to say the domain is Hits, and the image is on rank probability.
    """
    smoothPF=None
    """
    _`smoothPF` in a continuous TOC it stores an array of size `ndiscretization`_ that represents a ``smoothed`` or ``regularized`` derivative of the cummulatve (conditional to presence) distribution function. Notice that, often, the TOC curve coordinates are computed at spacing of 1 in the Hits plus False Alarms axis, and 0 or 1 in the Hits axis, this leads to noisy estimations of the probability functions, hence the discretization stored in `dHits`_ and `dHpFA`_ is intended for less noisy but precise representations that are suitable for fine-grain meaningful computations, nevertheless, these representations could be noisy for a visual analysis of the probability functions, hence, for visual analysis it is preferred to use the `smoothPF`_ representation, that is intended for a noise-free visualization, obviously there is a payoff of regularizing the TOC, hence we suggest to prefer `dHits`_ and `dHpFA`_ for computations unless  you are aware of the information lost.
    In a discrete TOC it stores a histogram.
    """
    smoothDPF=None
    """
    _`smoothDPF` is te first derivative of `smoothPF`_.
    """
    smoothingFactor=1
    """
    _`smoothingFactor` in a continuous TOC it is used in the `PFsmooth`_ method for the RLS method the greather smoothing factor the less smoothing.
    """

    icontinuous=None
    """
    _`icontinuous` is an array of size `ndiscretization`_ with 1 in the continuous segments and 0 in the discrete and -1 in the discontinuos (the segment is not in the domain of the rank).
    """

    iunique=None
    """
    _`iunique` is an array of size `ndata`_ with True in the last unique elements of the sorted rank and False otherwise. That is to say, if the sorted rank is  [0.3,0.3,0.3,0.5,0.7,1,1], iunique is [False,False,True,True,True,False,True].
    """
    featureName='X'
    """
    _`featureName` is the name of the feature to analyze for plotting purposes, it is 'X' by default.
    """
    boostrapFlag=False
    CImin=0.05
    CImax=0.95

    def __init__(self,rank=[], groundtruth=[], ndiscretization=-1, forceContinuity=True,smoothingMethod='ANN'):
        """
        _`__init__`
        Constructor of the TOC. Here the hits and false alarms are computed, as well as the kind (discrete, continuous, semicontinuous), the `Area`_  and `areaRatio`_ according to the definition in the documentation.
        """

        #validating rank, groundtruth pairs. They can never be 0 and unequal
        if (len(rank)!=0 and len(groundtruth)!=0 and len(rank)==len(groundtruth)):
            self.maxr=np.max(rank)
            """
            _`maxr` is the maximal rank value, it is a member of the TOCPF object.
            """
            self.maxgt=np.max(groundtruth)
            """
            _`maxgt` is the maximal (groundtruth) label value, it is usuaaly 1, it is a member of the TOCPF object.
            """
            self.minr=np.min(rank)
            
            """
            _`minr` is the minimal rank value, it is a member of the TOCPF object.
            """
            
            self.mingt=np.min(groundtruth)
            """
            _`mingt` is the minimal (groundtruth) label value, it is usuaaly 0, it is a member of the TOCPF object.
            """
            
            self.rank=rank
            """
            stores the rank values.
            """
            self.forceContinuity=forceContinuity
            self.groundtruth=groundtruth
            #Sorting the classification rank and getting the indices
            self.isorted=sorted(range(len(rank)),key=lambda index: rank[index],reverse=False)
            #Data size, this is the total number of samples
            self.ndata=len(rank)
            #This is the number of class 1 in the input data
            self.np=sum(groundtruth==1)
            #Hits plus false alarms
            self.HpFA=np.array(range(self.ndata))+1
            #True positives
            self.Hits=np.cumsum(groundtruth[self.isorted])
            #Thresholds
            self.Thresholds=rank[self.isorted]
            #Proportion of positives and data (positive class proportion)
            self.PDataProp=self.np/self.ndata
            #Detecting unique and repeated ranks/thresholds
            self.iunique=np.append(~((self.Thresholds[:-1]-self.Thresholds[1:])==0),True)
            self.sumIUnique=np.sum(self.iunique)
            #Computing the discretization for the TOC representation
            if (ndiscretization<0):
                self.discretization()
            else:
                self.ndiscretization=ndiscretization
            if (forceContinuity):
                self.ndiscretization=min(self.ndiscretization,np.sum(self.iunique))
            #Detecting continuous or discontinuous TOC.
            #self.continuity()
            self.areaComputation()
            self.computePF()

            # default parameters def PFsmoothing(self, method='RLS', PFsmoothingFactor=-1, 
            # DPFsmoothingFactor=-1, CDFsmoothingFactor=-1, dHitssmoothingFactor=-1 ): 
            self.PFsmoothing(smoothingMethod)
    pass



########################################Class definition ends########################################################


########################################BEGIN METHOD discretization##################################################

def discretization(self):
    """
    _`discretization` computes the number of segments to partition the rank domain, then it is used to determine whether the function is continuous or discontinuous.

    """
    self.ndiscretization=min(self.sumIUnique,min(min(int(self.ndata),10000),max(int(self.ndata/30),1000)))
    stopCond=int(min(self.sumIUnique,self.ndiscretization,self.ndata/self.ndiscretization))+1
    nCSegments=self.continuity()
    ite=0
    dfactor=0.5
    while(nCSegments<self.ndiscretization and (self.ndiscretization>stopCond and ite<25)):
        ite+=1
        #print(ite)
        self.ndiscretization=int(self.ndiscretization*(1-dfactor)+1)
        dfactor=max(0.1,dfactor/2)
        nCSegments=self.continuity()
    if (self.ndiscretization<=2):
        print('Can not find a continuous dicretization!!')
    else:
        self.areaDComputation()





########################################END METHOD discretization####################################################



########################################BEGIN METHOD continuity######################################################

def continuity(self):
    """
    _`continuity` computes the continuous, and discontinuous segments of the TOC.

    """
    thetaInf=self.minr-(0.5/self.ndata)*(self.maxr-self.minr)
    thetaSup=self.maxr+(0.5/self.ndata)*(self.maxr-self.minr)
    deltar=(thetaSup-thetaInf)/(self.ndiscretization)
    self.dThresholds=(np.array(range(self.ndiscretization))+1)*deltar+thetaInf
    j=0
    meanr=0
    sumHits=0
    lastSumHits=0
    lastSumHpFA=0
    sumHpFA=0
    nmean=0
    thetaSup=thetaInf+deltar
    self.icontinuous=np.zeros(self.ndiscretization)
    self.drank=np.zeros(self.ndiscretization)
    self.dHits=np.zeros(self.ndiscretization)
    self.dHpFA=np.zeros(self.ndiscretization)
    self.dndata=np.zeros(self.ndiscretization)
    self.icontinuous[0]=1
    for  i in range(self.ndata):
        testrank=self.rank[self.isorted[i]]
        while (testrank>=thetaSup):
            if (nmean>=1):
                self.icontinuous[j]=1
                self.drank[j]=meanr/nmean
                if (sumHits==0):
                    self.dHits[j]=lastSumHits
                    self.dHpFA[j]=lastSumHpFA
                self.dHits[j]=sumHits/nmean
                self.dHpFA[j]=sumHpFA/nmean
                self.dndata[j]=nmean
                lastSumHits=self.dHits[j]
                lastSumHpFA=self.dHpFA[j]
            else:
                if (self.forceContinuity):
                    self.drank[j]=(thetaInf+thetaSup)/2
                    self.dHits[j]=lastSumHits
                    self.dHpFA[j]=lastSumHpFA
                else:
                    self.drank[j]=float('nan')
                    self.dHits[j]=float('nan')
                    self.dHpFA[j]=float('nan')
            j+=1
            nmean=0
            meanr=0
            sumHits=0
            sumHpFA=0
            thetaInf=thetaSup
            thetaSup=thetaInf+deltar
        if (testrank>=thetaInf and testrank<thetaSup):
            meanr+=testrank
            nmean+=1
            sumHits+=self.Hits[i]
            sumHpFA+=self.HpFA[i]
    if (nmean>=1):
        self.icontinuous[j]=1
        self.drank[j]=meanr/nmean
        if (sumHits==0):
            self.dHits[j]=lastSumHits
        self.dHits[j]=sumHits/nmean
        self.dHpFA[j]=sumHpFA/nmean
        self.dndata[j]=nmean
        lastSumHits=self.dHits[j]
    else:
        if (self.forceContinuity):
            self.drank[j]=(thetaInf+thetaSup)/2
            self.dHits[j]=lastSumHits
        else:
            self.drank[j]=float('nan')
            self.dHits[j]=float('nan')
            self.dHpFA[j]=float('nan')

    continuousSegments=np.sum(self.icontinuous)
    if (continuousSegments==(self.ndiscretization)):
        self.kind='continuous'
    elif (continuousSegments>=int(self.ndiscretization/2)): #If there are some gaps in the function
        self.kind='semicontinuous'
        if (self.forceContinuity): #If there are some gaps in the function
            self.kind='forcedContinuous'
    else:
        self.kind='discrete'
    if ((self.sumIUnique)<(self.ndata/2) and self.kind!='continuous'):
        self.kind='discrete'
    if (self.kind=='discrete'):
        self.drank=self.Thresholds[self.iunique]
        self.dHits=self.Hits[self.iunique]
        self.dHpFA=self.HpFA[self.iunique]

    return continuousSegments



def boostrapContinuity(self,rank,Hits,HpFA,isorted):
    """
    _`continuity` computes the continuous, and discontinuous segments of the TOC.

    """
    minr=np.min(rank)
    maxr=np.max(rank)
    thetaInf=minr-(0.5/self.ndata)*(maxr-minr)
    thetaSup=maxr+(0.5/self.ndata)*(maxr-minr)
    deltar=(thetaSup-thetaInf)/(self.ndiscretization)
    j=0
    meanr=0
    sumHits=0
    lastSumHits=0
    lastSumHpFA=0
    sumHpFA=0
    nmean=0
    thetaSup=thetaInf+deltar
    drank=np.zeros(self.ndiscretization)
    dHits=np.zeros(self.ndiscretization)
    dHpFA=np.zeros(self.ndiscretization)
    Thresholds=rank[isorted]
    iunique=np.append(~((Thresholds[:-1]-Thresholds[1:])==0),True)
    if (self.kind=='discrete'):
        #Detecting unique and repeated ranks/thresholds
        drank=Thresholds[iunique]
        dHits=Hits[iunique]
        dHpFA=HpFA[iunique]
        return drank,dHits,dHpFA, iunique

    for  i in range(self.ndata):
        testrank=rank[isorted[i]]
        while (testrank>=thetaSup):
            if (nmean>=1):
                drank[j]=meanr/nmean
                if (sumHits==0):
                    dHits[j]=lastSumHits
                    dHpFA[j]=lastSumHpFA
                dHits[j]=sumHits/nmean
                dHpFA[j]=sumHpFA/nmean
                lastSumHits=dHits[j]
                lastSumHpFA=dHpFA[j]
            else:
                drank[j]=(thetaInf+thetaSup)/2
                dHits[j]=lastSumHits
                dHpFA[j]=lastSumHpFA
            j+=1
            nmean=0
            meanr=0
            sumHits=0
            sumHpFA=0
            thetaInf=thetaSup
            thetaSup=thetaInf+deltar
        if (testrank>=thetaInf and testrank<thetaSup):
            meanr+=testrank
            nmean+=1
            sumHits+=Hits[i]
            sumHpFA+=HpFA[i]

    if (nmean>=1):
        drank[j]=meanr/nmean
        if (sumHits==0):
            dHits[j]=lastSumHits
        dHits[j]=sumHits/nmean
        dHpFA[j]=sumHpFA/nmean
        lastSumHits=dHits[j]
    else:
        drank[j]=(thetaInf+thetaSup)/2
        dHits[j]=lastSumHits

    return drank,dHits,dHpFA, iunique



##########################################END METHOD continuity#####################################################

########################################BEGIN METHOD computePF######################################################


def centeredDF(self,n,X,Y): #centered finite differeces for derivatives
    """
    _`centeredDF`Computes derivatives using centered finite differences.
    """
    #X=X[self.icontinuous.astype(bool)]
    #Y=Y[self.icontinuous.astype(bool)]
    #n=len(X)

    DX=np.zeros(n)
    n=len(X)
    DX[1:(n-2)]=0.5*(Y[1:(n-2)]-Y[:(n-3)])/(X[1:(n-2)]-X[:(n-3)])+0.5*(Y[2:(n-1)]-Y[1:(n-2)])/(X[2:(n-1)]-X[1:(n-2)])
    DX[0]=(Y[1]-Y[0])/(X[1]-X[0])
    DX[-1]=(Y[-1]-Y[-2])/(X[-1]-X[-2])
    return DX


def integrateTrapezoidal(self,n,X,Y,Reference,zerofix): #centered finite differeces for derivatives
    """
    _`integrateTrapezoidal` Computes integrals using the trapezoidal rule.
    """
    IX=np.zeros(n)
    IX[1:]=(Y[:(n-1)]+Y[1:])*(X[1:]-X[:(n-1)])
    IX[0]=0
    #IX=np.cumsum(IX)
    if (len(Reference)>0):
        maxr=np.max(Reference)
        minr=np.min(Reference)
        maxi=np.max(IX)
        mini=np.min(IX)
        IX=0.99*(maxr-minr)/(maxi-mini)*IX
        IX=IX+(np.mean(Reference)-np.mean(IX) )
    if(zerofix):
        IX[IX<0]=0
    IX=IX+(np.mean(Reference)-np.mean(IX) )

    return IX


def computePF(self,method='centeredDF'):
    """

    This method scales the Hits axis to the interval [0,1]. The self TOC (the TOC which the method is called from) is normalized, there is not new memory allocation.
    :return: Returns the modified TOC curve
    :rtype: ``TOC``

    The ``kind`` TOC curve is *'normalized'*.
    The  true positives plus false positives count is 1, ntppfp=1,
    and true positives, TP=1.
    Nevertheless the basentppfp and basenpos stores the values of the self TOC.

    """
    self.CDF=np.zeros(self.ndiscretization)
    self.CDF=self.dHits/self.np
    if (self.kind=='continuous'):
        self.PF=self.centeredDF(self.ndiscretization,self.drank,self.CDF)
        self.DPF=self.centeredDF(self.ndiscretization,self.drank,self.PF)
    if (self.kind=='discrete'):
        self.PF=np.zeros(self.ndiscretization)
        self.DPF=np.zeros(self.ndiscretization)
        self.PF[1:]=self.CDF[1:]-self.CDF[:-1]
        self.PF[0]=self.CDF[0]
        self.DPF[1:]=self.PF[1:]-self.PF[:-1]
        self.DPF[0]=self.PF[0]
    return self.PF


def boostrapComputePF(self,dHits,drank,npres,method='centeredDF'):
    """

    This method scales the Hits axis to the interval [0,1]. The self TOC (the TOC which the method is called from) is normalized, there is not new memory allocation.
    :return: Returns the modified TOC curve
    :rtype: ``TOC``

    The ``kind`` TOC curve is *'normalized'*.
    The  true positives plus false positives count is 1, ntppfp=1,
    and true positives, TP=1.
    Nevertheless the basentppfp and basenpos stores the values of the self TOC.

    """
    ndiscretization=len(dHits)
    CDF=np.zeros(ndiscretization)
    CDF=dHits/npres
    if (self.kind=='continuous'):
        PF=self.centeredDF(ndiscretization,drank,CDF)
        DPF=self.centeredDF(ndiscretization,drank,PF)
    if (self.kind=='discrete'):
        PF=np.zeros(ndiscretization)
        DPF=np.zeros(ndiscretization)
        PF[1:]=CDF[1:]-CDF[:-1]
        PF[0]=CDF[0]
        DPF[1:]=PF[1:]-PF[:-1]
        DPF[0]=PF[0]
    return CDF,PF,DPF



def bestPercent(self,X,PF,n, percent):
    imax=np.argmax(PF)
    mass=0
    i1=0
    i0=0
    xareas=np.zeros(n)
    xareas[1:(n-1)]=(X[1:]-X[:-1])/2+(X[2:(n)]-X[1:-1])/2
    xareas[0]=(X[1]-X[0])/2
    xareas[-1]=(X[-1]-X[-2])/2
    ima=imax
    imi=imax
    xareas=PF*xareas
    totalMass=np.sum(xareas)
    massCentered=np.sum(xareas[imi:ima])
    for i in range(n):
        ima=imax+i1
        imi=imax-i0
        print('massCentered',massCentered)
        print('ima',ima)
        print('imi',imi)
        if ((imi-1)>-1):
            massLeft=massCentered+xareas[(imi-1)]
            i0=1
        if ((ima+1)<n):
            massRight=massCentered+xareas[(ima+1)]
            i1=1
        if (massLeft>=massRight):
            imi=imi-i0
            massCentered=massLeft
            i0=imax-imi
        else:
            ima=ima+i1
            i1=ima-imax
            massCentered=massRight
    return imax,i0,i1



########################################END METHOD computePF####################################################

def PFsmoothing(self, method='ANN', PFsmoothingFactor=-1, DPFsmoothingFactor=-1, CDFsmoothingFactor=-1, dHitssmoothingFactor=-1 ):
    # Aqui podemos implementar recepción de argumentos variables con kwargs o un diccionario https://python-intermedio.readthedocs.io/es/latest/args_and_kwargs.html
    

    if (method == 'RLS'):
        if (dHitssmoothingFactor==-1):
            dHitssmoothingFactor=self.smoothingFactor*2
        if (CDFsmoothingFactor==-1):
            CDFsmoothingFactor=self.smoothingFactor*2
        if (PFsmoothingFactor==-1):
            PFsmoothingFactor=self.smoothingFactor
        if (DPFsmoothingFactor==-1):
            DPFsmoothingFactor=10.0*self.smoothingFactor
        self.smoothdHits=self.RLS(self.drank,self.dHits,self.ndiscretization,dHitssmoothingFactor)
        self.smoothCDF=self.RLS(self.drank,self.CDF,self.ndiscretization,CDFsmoothingFactor)
        self.smoothPF=self.RLS(self.drank,self.PF,self.ndiscretization,PFsmoothingFactor)
        self.smoothDPF=self.RLS(self.drank,self.DPF,self.ndiscretization,DPFsmoothingFactor)
    elif (method=='wmeans'):
        self.smoothdHits=self.meanWindowSmoothing(self.drank,self.dHits,self.ndiscretization,-1)
        self.smoothCDF=self.meanWindowSmoothing(self.drank,self.CDF,self.ndiscretization,-1)
        self.smoothPF=self.meanWindowSmoothing(self.drank,self.PF,self.ndiscretization,-1)
        def mfunc(x):
            return x[np.argmax(np.abs(x))]
        self.smoothDPF=self.meanWindowSmoothing(self.drank,self.DPF,self.ndiscretization,-1,mfunc)
    
    elif (method == "ANN"):
        """
        Artificial Neural Networks
        """
        if (self.kind=='continuous'):
            X, Y, DY, DDY = self.fitNN(self.drank, self.dHits)
            maxY=np.max(np.array(Y))
            self.smoothdHits = np.array(Y)
            self.smoothCDF = np.array(Y)/maxY
            integ=(np.mean(np.array(DY))*(np.max(self.drank) - np.min(self.drank)))
            self.smoothPF = np.array(DY)/integ
            #(Y[:(n-1)]+Y[1:])*(X[1:]-X[:(n-1)])/2
            self.smoothDPF = np.array(DDY)/integ
        else:
            self.smoothdHits=self.meanWindowSmoothing(self.drank,self.dHits,self.ndiscretization,-1)
            self.smoothCDF=self.meanWindowSmoothing(self.drank,self.CDF,self.ndiscretization,-1)
            self.smoothPF=self.meanWindowSmoothing(self.drank,self.PF,self.ndiscretization,-1)
            integ=np.sum(self.smoothPF)
            self.smoothPF=self.smoothPF/integ
            def mfunc(x):
                return x[np.argmax(np.abs(x))]
            self.smoothDPF=self.meanWindowSmoothing(self.drank,self.DPF,self.ndiscretization,-1,mfunc)

    else:
        print("method=",method,"is not implemented!")



def boostrapPFsmoothing(self,drank,dHits,CDF,PF,DPF,PFsmoothingFactor=-1,DPFsmoothingFactor=-1,CDFsmoothingFactor=-1,dHitssmoothingFactor=-1,method='RLS' ):

    if (method=='RLS'):
        if (dHitssmoothingFactor==-1):
            dHitssmoothingFactor=self.smoothingFactor*2
        #self.smoothdHits=self.RLS(self.dHpFA,self.dHits,self.ndiscretization,dHitssmoothingFactor)
        if (CDFsmoothingFactor==-1):
            CDFsmoothingFactor=self.smoothingFactor*2
        if (PFsmoothingFactor==-1):
            PFsmoothingFactor=self.smoothingFactor
        if (DPFsmoothingFactor==-1):
            DPFsmoothingFactor=10.0*self.smoothingFactor
            #print("Entra en el if")
        ndiscretization=len(dHits)
        smoothdHits=self.RLS(drank,dHits,ndiscretization,dHitssmoothingFactor)
        smoothCDF=self.RLS(drank,CDF,ndiscretization,CDFsmoothingFactor)
        smoothPF=self.RLS(drank,PF,ndiscretization,PFsmoothingFactor)
        smoothDPF=self.RLS(drank,DPF,ndiscretization,DPFsmoothingFactor)
    else:
        print("method=",method,"is not implemented!")
    return smoothdHits,smoothCDF,smoothPF,smoothDPF


def meanWindowSmoothing(self,X,Y,n,smoothingFactor=-1,mfunction=np.mean):
    Yg=np.zeros(n)
    #Smoothing the density using a mean filter, it is similar to a uniform kernel with a window size=smoothing
    if (smoothingFactor==-1):
        nw=min(int(n/2),200)
        smoothing=int(n/nw)
        #density.smwindow=smoothing
    if (smoothing>0):
        Yg[0:smoothing]= mfunction(Y[0:smoothing])
        Yg[(n-smoothing):n]=  mfunction(Y[(n-smoothing):n])
        for i in range(smoothing,n-smoothing):
            Yg[i]=mfunction(Y[(i-smoothing):(i+smoothing)])
    return Yg


def RLS(self, X, Y, n, smoothingFactor=-1):
    """
    
    """
    mixx=np.min(X)
    maxx=np.max(X)
    miyy=np.min(Y)
    mayy=np.max(Y)
    active=np.zeros(n)
    if (smoothingFactor==-1):
        smoothingFactor=self.smoothingFactor
    dx=self.centeredDF(n,(X-mixx)*n/(maxx-mixx),(Y-miyy)/(mayy-miyy))
    Yg=np.copy(Y)
    l=1/smoothingFactor**2
    smoothingFactor=smoothingFactor*250/(n**1.4)
    active[1:-1]=np.logical_or( np.abs(dx[1:-1]-dx[2:])>smoothingFactor, np.abs(dx[0:-2]-dx[1:-1])>smoothingFactor).astype(float)

    nactive=1 # np.sum(active)
    maxdx=np.max((np.abs(dx[1:-1]-dx[2:]))**2+(np.abs(dx[0:-2]-dx[1:-1]))**2)
    maxy=np.max(np.abs(Y))
    active[1:-1]=active[1:-1]*(1-(((np.abs(dx[1:-1]-dx[2:]))**2+(np.abs(dx[0:-2]-dx[1:-1]))**2)/(maxdx+1.0e-9)))
    active[active==0]=1
    smoothingIte=0
    while(nactive>0 and l<9e6):
        #print('nactive', nactive)
        mat=np.zeros((3,n))
        #diagonal
        #mat[1,1:(n-1)]=wdiag[1:(n-1)]+2*l*active[1:(n-1)]
        #mat[1,0]=wdiag[0]+1*l*active[0]
        #mat[1,n-1]=wdiag[n-1]+1*l*active[n-1]
        mat[1,1:(n-1)]=1+2*l*active[1:(n-1)]
        mat[1,0]=1+1*l*active[0]
        mat[1,n-1]=1+1*l*active[n-1]
        #Equations from 1 to n-1, starting at 0
        mat[0,1:]=-l*active[:-1]
        #Equations from 1 to n-2, starting at 0
        mat[2,0:(n-1)]=-l*active[1:]
        b=Y
        Yg=la.solve_banded ((1,1),mat,b)
        dx=self.centeredDF(n,(X-mixx)*n/(maxx-mixx),(Yg-miyy)/(mayy-miyy))
        activefactor=nactive
        nactive=np.sum(np.logical_or(np.abs(dx[1:-1]-dx[2:])>smoothingFactor, np.abs(dx[0:-2]-dx[1:-1])>smoothingFactor).astype(float))
        l=(1+0.5*nactive/(activefactor+1))*l
        smoothingIte+=1
    self.smoothingIte=smoothingIte
    self.smoothingLambda=l
        #print('l',l)
    areaYg=np.sum(np.abs(0.5*(Yg[:-1]+Yg[1:])*(X[1:]-X[:-1])))
    areaY=np.sum(np.abs(0.5*(Y[:-1]+Y[1:])*(X[1:]-X[:-1])))
    ite=0
    while(abs(areaYg-1)>1e-4 and ite<10):
        Yg=areaY/(areaYg+1e-8)*Yg
        ite+=1
        areaYg=np.sum(np.abs(0.5*(Yg[:-1]+Yg[1:])*(X[1:]-X[:-1])))
    return Yg




def fitNN(self, X, Y, structure = [25, 25, 25], afunctions = ["sigmoid", "sigmoid", "sigmoid"]):

    #intepolacion lineal para tener valores en cada TPpFP
    #0,5,10
    #X = 0,1,2,3,4,5,6,7,8,9,10
    #sigmoidal expand tails for reinforce learning at those extremes
    #stail = 1000000
    maxX0=np.max(X)
    maxY0=np.max(Y)
    minX0=np.min(X)
    #Xinterp = np.arange(0, len(X))
    Xinterp = ((np.arange(0, len(X)+1))*(maxX0-minX0)/(len(X)))+minX0
    Yinterp = np.interp(Xinterp, X, Y)
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
        # Stop training when `val_loss` is no longer improving
        monitor="val_loss",
        # "no longer improving" being defined as "no better than 1e-2 less"
        min_delta=1e-7,
        # "no longer improving" being further defined as "for at least 10 epochs"
        patience = 100,
        verbose=1,
        )
    ]

    Dx0=Xinterp[1]-Xinterp[0]
    Dy0=Yinterp[1]-Yinterp[0]
    Dx1=Xinterp[-1]-Xinterp[-2]
    Dy1=Yinterp[-1]-Yinterp[-2]
    X_train_e=Xinterp
    Y_train_e=Yinterp
    for i in range(max(int(0.05*len(Xinterp)),3)):
        X_train_e = np.append(Xinterp[0]-float(i)*Dx0,X_train_e)
        Y_train_e = np.append(Yinterp[0]-float(i)*Dy0,Y_train_e)
        X_train_e = np.append(X_train_e ,X_train_e[-1]+float(i)*Dx1)
        Y_train_e = np.append(Y_train_e ,Y_train_e[-1]+float(i)*Dy1)

    self.ntrain=len(X_train_e)
    maxX=np.max(X_train_e)
    maxY=np.max(Y_train_e)
    minX=np.min(X_train_e)
    minY=np.min(Y_train_e)

    #Normalized TOC
    X_train = ((X_train_e-minX)/(maxX-minX))
    Y_train = ((Y_train_e-minY)/(maxY-minY))

    X_train_s = X_train[::2]
    Y_train_s = Y_train[::2]

    X_valid_s = X_train[1:][::2]
    Y_valid_s = Y_train[1:][::2]

    model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(1,)),        # Input layer of size 1
    tf.keras.layers.Dense(structure[0], input_dim = 1, activation= afunctions[0]),
    tf.keras.layers.Dense(structure[1], input_dim = structure[0], activation=afunctions[1]),
    tf.keras.layers.Dense(structure[2], input_dim = structure[1], activation= afunctions[2]),
    tf.keras.layers.Dense(1, input_dim = structure[2], activation='linear')
    ])

        # Compile the model
    model.compile(optimizer = 'adam',  loss = "mse")

    # Train the model
    model.fit(X_train_s, Y_train_s, validation_data=(X_valid_s, Y_valid_s), epochs=10000, callbacks=callbacks, verbose=1, batch_size=int(len(X_train_s)/10))
    #model.fit(X_train, Y_train, validation_data=(X_train, Y_train), epochs=10000, callbacks=callbacks, verbose=1, batch_size=int(len(X_train)/10))


    #regresar la predicción el dominio en las coordenadas de X

    yhat = model.predict( (X-minX)/(maxX-minX))
    realHits = (yhat)*(maxY-minY)+minY
    #print(realHits)
    input_data = tf.convert_to_tensor((X-minX)/(maxX-minX))
    #output = tf.convert_to_tensor(Y*(maxY-minY)+minY)
    with tf.GradientTape() as tape2:
        tape2.watch(input_data)
        with tf.GradientTape() as tape1:
            tape1.watch(input_data)
            output = model(input_data)
        first_derivative = tape1.gradient(output, input_data)
    second_derivative = tape2.gradient(first_derivative, input_data)


    return np.array(X), np.array(realHits), np.array(first_derivative), np.array(second_derivative)



##########################################BEGIN METHOD areaComputacion###############################################

def areaComputation(self):
    """

    This method computes the areaun der the curve of the TOC and parallelogram and the proportional ratio
    :return: Returns the TOC's area under the curve
    :rtype: ``float``

    """
    self.area=0
    AUC=0
    pararea=0
    if (self.kind=='discrete'):
        #Th equivalent area under the curve is the sum of hits
        AUC=np.sum(self.Hits[self.iunique])
        #Sum of parallelogram heights in the first section
        idx=self.HpFA[self.iunique]<self.np
        parareaBP=np.sum(self.HpFA[self.iunique][idx])

        #Sum of parallelogram heights in the second section
        idx=self.HpFA[self.iunique]>=self.np
        parareaP=self.np*np.sum(idx)

        #Sum of parallelogram heights in the last  section
        xoffset=self.ndata-self.np
        idx=self.HpFA[self.iunique]>xoffset
        parareaAP=np.sum((self.HpFA[self.iunique][idx]-xoffset))
        #This is th equivalent area under the curve miinus the last triangle of the paralellogram
        self.area=AUC-parareaAP
        pararea=parareaBP+parareaP-parareaAP
    else:
        AUC=np.sum(0.5*(self.Hits[self.iunique][:-1]+self.Hits[self.iunique][1:])*(self.HpFA[self.iunique][1:]-self.HpFA[self.iunique][:-1]))
        pararea=self.np*self.ndata
        self.area=AUC-(self.np*self.np)/2
    self.areaRatio=self.area/pararea
    self.pararea=pararea
    return AUC


##########################################END METHOD areaComputacion#################################################



def areaComputationBoostrap(self,Hits,HpFA,iunique,npres,ndata):
    """

    This method computes the area under the curve of the TOC and parallelogram and the proportional ratio
    :return: Returns the TOC's area under the curve
    :rtype: ``float``

    """
    area=0
    AUC=0
    pararea=0
    if (self.kind=='discrete'):
        #The equivalent area under the curve is the sum of hits
        AUC=np.sum(Hits[iunique])
        #Sum of parallelogram heights in the first section
        idx=HpFA[iunique]<npres
        parareaBP=np.sum(HpFA[iunique][idx])

        #Sum of parallelogram heights in the second section
        idx=HpFA[iunique]>=npres
        parareaP=npres*np.sum(idx)

        #Sum of parallelogram heights in the last  section
        xoffset=ndata-npres
        idx=HpFA[iunique]>xoffset
        parareaAP=np.sum((HpFA[iunique][idx]-xoffset))
        #This is th equivalent area under the curve miinus the last triangle of the paralellogram
        area=AUC-parareaAP
        pararea=parareaBP+parareaP-parareaAP
    else:
        AUC=np.sum(0.5*(Hits[iunique][:-1]+Hits[iunique][1:])*(HpFA[iunique][1:]-HpFA[iunique][:-1]))
        pararea=npres*ndata
        area=AUC-(npres*npres)/2
    areaRatio=area/pararea
    pararea=pararea
    return areaRatio


##########################################END METHOD areaComputacion#################################################



##########################################BEGIN METHOD areaComputacion###############################################

def areaDComputation(self):
    """

    This method computes the areaun der the curve of the TOC and parallelogram and the proportional ratio
    :return: Returns the TOC's area under the curve
    :rtype: ``float``

    """
    area=0
    AUC=0
    pararea=0
    ##########!!!!!!!!!!ME FALTA PROBAR EL CASO DISCRETO############
    if (self.kind=='discrete'):
        #Th equivalent area under the curve is the sum of hits
        AUC=np.sum(self.dHits)
        #Sum of parallelogram heights in the first section
        idx=self.dHpFA<self.np
        parareaBP=np.sum(self.dHpFA[idx])

        #Sum of parallelogram heights in the second section
        idx=self.dHpFA>=self.np
        parareaP=self.np*np.sum(idx)

        #Sum of parallelogram heights in the last  section
        xoffset=self.ndata-self.np
        idx=self.dHpFA>xoffset
        parareaAP=np.sum((self.dHpFA[idx]-xoffset))
        #This is th equivalent area under the curve miinus the last triangle of the paralellogram
        self.Darea=AUC-parareaAP
        pararea=parareaBP+parareaP-parareaAP
        #self.areaRatio=area/pararea
    else:
        nonan=~np.isnan(self.dHits)
        AUC=np.sum(0.5*(self.dHits[nonan][:-1]+self.dHits[nonan][1:])*(self.dHpFA[nonan][1:]-self.dHpFA[nonan][:-1]))
        pararea=self.np*self.ndata
        self.Darea=AUC-(self.np*self.np)/2
    self.areaDRatio=self.Darea/pararea
    pararea=pararea
    #print('AreaRatio', self.areaDRatio)
    #print('pararea', pararea)
    return self.areaDRatio



##########################################END METHOD areaComputacion#################################################


#This function computes a probability value given

def rank2prob(self,rank,kind='PF'):

    """

    This function computes probability values associated to a rank value. The ``thresholds`` array of the density TOC is used for this purpose.
    Very possibly the rank is the same than those used to compute a standard TOC instantiated by TOC(rank,groundtruth), hence the indices used
    in the constructor are available and can save computational time. Otherwise the indices are recomputed. In any case the inputed ``rank``
    array must be in the same interval than the thresholds.

    :param rank: A numpy array with the ran. The intended uses is that this rank comes from the standard TOC computation, and to associate this rank with geolocations, hence the probabilities can be associated in the same order.

    :param kind: if ``kind`` is ''density'' the probabilitites are computed with the non-smoothed density function, otherwise the smooth version is used. Notice that a this function only must be called from a ''density'' kind TOC. Optional

    :param indices: Indices of the reversely sorted rank, this array is computed by the standard TOC computation. Hence the computational cost of recomputing them could be avoided, otherwise the indices are recomputed and they are not stored. Optional

    return: a numpy array with the probabilities. The probabilities do not sum 1, instead they sum ``PDataProp``, that is to say they sum the proportion of positives in the data. That is an estimation of the probability of having a 1-class valued datum.

    :rtype: numpy array

    """
    indices=sorted(range(len(rank)),key=lambda index: rank[index],reverse=False)
    if (kind=='PF'):
        DF=self.PF
    elif(kind=='dHits'):
        DF=self.dHits
    elif(kind=='CDF'):
        DF=self.CDF
    elif(kind=='DPF'):
        DF=self.DPF
    elif(kind=='smoothPF'):
        DF=self.smoothPF
    elif(kind=='smoothDPF'):
        DF=self.smoothDPF

    nr=len(rank)
    nd=self.ndiscretization
    prob=np.zeros(nr)
    j=0
    deltammr=0.5*(self.maxr-self.minr)/self.ndiscretization
    for i in range(nd-1):
        nd=0
        jini=j
        while(j<nr  and  (rank[indices[j]])<=(self.drank[i+1])):
            if (rank[indices[j]]>=self.minr and rank[indices[j]]<=self.maxr):
                deltar=self.drank[i]-self.drank[i-1]
                nd=nd+1
                prob[indices[j]]=  (DF[i+1]*(rank[indices[j]]-self.drank[i])+DF[i]*(self.drank[i+1]-rank[indices[j]]))/deltar
            elif(rank[indices[j]]<self.minr and rank[indices[j]]>=(self.minr-deltammr)):
                prob[indices[j]]=DF[0]
                nd=nd+1
            elif(rank[indices[j]]>self.maxr and rank[indices[j]]<=(self.maxr+deltammr)):
                prob[indices[j]]=DF[-1]
                nd=nd+1
            else:
                prob[indices[j]]=0
            j+=1
            jend=j
        if (jini!=jend):
            prob[indices[jini:jend]]=prob[indices[jini:jend]]/(jend-jini)
    return(prob)


def simulate(self,rank,nprop=1):
    prob=self.rank2prob(rank)
    prob=np.cumsum(prob)
    prob=prob/prob[-1]
    #print(prob[-1])
    simulation=np.zeros(len(rank))
    if (nprop<1):
        nprop=int(nprop*len(prob)+0.5)
    else:
        nprop=int(nprop)
    nsim=0
    #while(nsim<nprop):
    rsamples=sorted(np.random.rand(nprop-nsim))
    j=0
    for i in range(len(prob)):
        while(rsamples[j]<prob[i] and j<len(rsamples)):
            simulation[i]=1
            j+=1
            if (j==len(rsamples)):
                break
        if (j==len(rsamples)):
            break
    nsim=int(np.sum(simulation))
        #print('nsim',nsim,'nprop',nprop)
    return simulation





def boostrapCI(self,nboostrap=100,CImin=0.05,CImax=0.95):

    areas=np.zeros(nboostrap)
    for i in range(nboostrap):
        state=np.random.get_state()
        #print(i, np.random.rand(1))
        bindex=np.random.randint(0, high=self.ndata-1, size=self.ndata)
        rank=self.rank[bindex]
        groundtruth=self.groundtruth[bindex]
        isorted=sorted(range(len(rank)),key=lambda index: rank[index],reverse=False)
        #True positives
        Hits=np.cumsum(groundtruth[isorted])
        #Detecting unique and repeated ranks/thresholds
        iunique=np.append(~((rank[isorted][:-1]-rank[isorted][1:])==0),True)
        areas[i]=self.areaComputationBoostrap(Hits,self.HpFA,iunique,np.sum(groundtruth),self.ndata)
        if (i==0):
            state0=state
        if (i==int(nboostrap/4+0.5)):
            state1=state
        if (i==int(nboostrap/2+0.5)):
            state2=state
        if (i==int(3*nboostrap/4+0.5)):
            state3=state

    isorted=sorted(range(len(areas)),key=lambda index: areas[index],reverse=False)
    self.boostrapAreas=areas[isorted]
    imin=int(CImin*nboostrap+0.5)
    imax=int(CImax*nboostrap+0.5)
    iCImin=isorted[imin]
    iCImax=isorted[imax]
    #print(iCImin)
    #print(iCImax)
    stateMin=state3
    stateMax=state3
    irandmin=int(3*nboostrap/4+0.5)
    irandmax=int(3*nboostrap/4+0.5)
    if (iCImin<irandmin):
        stateMin=state2
        irandmin=int(nboostrap/2+0.5)
    if (iCImin<int(nboostrap/2+0.5)):
        stateMin=state1
        irandmin=int(nboostrap/4+0.5)
    if (iCImin<int(nboostrap/4+0.5)):
        stateMin=state0
        irandmin=0

    if (iCImax<irandmax):
        stateMax=state2
        irandmax=int(nboostrap/2+0.5)
    if (iCImax<int(nboostrap/2+0.5)):
        stateMax=state1
        irandmax=int(nboostrap/4+0.5)
    if (iCImax<int(nboostrap/4+0.5)):
        stateMax=state0
        irandmax=0
    flag=False
    if (irandmin==irandmax):
        np.random.set_state(stateMin)
        #print(np.random.rand(1))
        for i in range(irandmin,nboostrap):
            bindex=np.random.randint(0, high=self.ndata-1, size=self.ndata)
            rank=self.rank[bindex]
            groundtruth=self.groundtruth[bindex]
            if (iCImin==i):
                isorted=sorted(range(len(rank)),key=lambda index: rank[index],reverse=False)
                Hits=np.cumsum(groundtruth[isorted])
                iunique=np.append(~((rank[isorted][:-1]-rank[isorted][1:])==0),True)
                self.CIminNpres=np.sum(groundtruth)
                self.CIminArea=self.areaComputationBoostrap(Hits,self.HpFA,iunique,self.CIminNpres,self.ndata)
                self.CIminRank=rank[isorted]
                self.CIminHits=Hits
                self.CIminIUnique=iunique
                self.CIminDrank,self.CIminDHits,self.CIminDHpFA,self.CIminIUnique=self.boostrapContinuity(rank,Hits,self.HpFA,isorted)
                self.CIminCDF,self.CIminPF,self.CIminDPF=self.boostrapComputePF(self.CIminDHits,self.CIminDrank,self.CIminNpres,'centeredDF')
                self.CIminSmoothdHits,self.CIminSmoothCDF,self.CIminSmoothPF,self.CIminSmoothDPF=self.boostrapPFsmoothing(self.CIminDrank,self.CIminDHits,self.CIminCDF,self.CIminPF,self.CIminDPF)
                if (flag==True):
                    break
                flag=True
            if (iCImax==i):
                isorted=sorted(range(len(rank)),key=lambda index: rank[index],reverse=False)
                Hits=np.cumsum(groundtruth[isorted])
                iunique=np.append(~((rank[isorted][:-1]-rank[isorted][1:])==0),True)
                self.CImaxNpres=np.sum(groundtruth)
                self.CImaxArea=self.areaComputationBoostrap(Hits,self.HpFA,iunique,self.CImaxNpres,self.ndata)
                self.CImaxRank=rank[isorted]
                self.CImaxHits=Hits
                self.CImaxIUnique=iunique
                self.CImaxDrank,self.CImaxDHits,self.CImaxDHpFA,self.CIminIUnique=self.boostrapContinuity(rank,Hits,self.HpFA,isorted)
                self.CImaxCDF,self.CImaxPF,self.CImaxDPF=self.boostrapComputePF(self.CImaxDHits,self.CImaxDrank,self.CImaxNpres,'centeredDF')
                self.CImaxSmoothdHits,self.CImaxSmoothCDF,self.CImaxSmoothPF,self.CImaxSmoothDPF=self.boostrapPFsmoothing(self.CImaxDrank,self.CImaxDHits,self.CImaxCDF,self.CImaxPF,self.CImaxDPF)
                if (flag==True):
                    break
                flag=True
    else:
        np.random.set_state(stateMin)
        #print(np.random.rand(1))
        for i in range(irandmin,nboostrap):
            bindex=np.random.randint(0, high=self.ndata-1, size=self.ndata)
            rank=self.rank[bindex]
            groundtruth=self.groundtruth[bindex]
            if (iCImin==i):
                isorted=sorted(range(len(rank)),key=lambda index: rank[index],reverse=False)
                Hits=np.cumsum(groundtruth[isorted])
                iunique=np.append(~((rank[isorted][:-1]-rank[isorted][1:])==0),True)
                self.CIminNpres=np.sum(groundtruth)
                self.CIminArea=self.areaComputationBoostrap(Hits,self.HpFA,iunique,self.CIminNpres,self.ndata)
                self.CIminRank=rank[isorted]
                self.CIminHits=Hits
                self.CIminIUnique=iunique
                self.CIminDrank,self.CIminDHits,self.CIminDHpFA,self.CIminIUnique=self.boostrapContinuity(rank,Hits,self.HpFA,isorted)
                self.CIminCDF,self.CIminPF,self.CIminDPF=self.boostrapComputePF(self.CIminDHits,self.CIminDrank,self.CIminNpres,'centeredDF')
                self.CIminSmoothdHits,self.CIminSmoothCDF,self.CIminSmoothPF,self.CIminSmoothDPF=self.boostrapPFsmoothing(self.CIminDrank,self.CIminDHits,self.CIminCDF,self.CIminPF,self.CIminDPF)
                break
        np.random.set_state(stateMax)
        #print(np.random.rand(1))
        for i in range(irandmax,nboostrap):
            bindex=np.random.randint(0, high=self.ndata-1, size=self.ndata)
            rank=self.rank[bindex]
            groundtruth=self.groundtruth[bindex]
            if (iCImax==i):
                isorted=sorted(range(len(rank)),key=lambda index: rank[index],reverse=False)
                Hits=np.cumsum(groundtruth[isorted])
                iunique=np.append(~((rank[isorted][:-1]-rank[isorted][1:])==0),True)
                self.CImaxNpres=np.sum(groundtruth)
                self.CImaxArea=self.areaComputationBoostrap(Hits,self.HpFA,iunique,self.CImaxNpres,self.ndata)
                self.CImaxRank=rank[isorted]
                self.CImaxHits=Hits
                self.CImaxIUnique=iunique
                self.CImaxDrank,self.CImaxDHits,self.CImaxDHpFA,self.CIminIUnique=self.boostrapContinuity(rank,Hits,self.HpFA,isorted)
                self.CImaxCDF,self.CImaxPF,self.CImaxDPF=self.boostrapComputePF(self.CImaxDHits,self.CImaxDrank,self.CImaxNpres,'centeredDF')
                self.CImaxSmoothdHits,self.CImaxSmoothCDF,self.CImaxSmoothPF,self.CImaxSmoothDPF=self.boostrapPFsmoothing(self.CImaxDrank,self.CImaxDHits,self.CImaxCDF,self.CImaxPF,self.CImaxDPF)
                break
    self.CImin=CImin
    self.CImax=CImax
    self.boostrapFlag=True
    lcimin=len(self.CIminDrank)
    lcimax=len(self.CImaxDrank)
    if (lcimin<lcimax):
        self.CIminIndex=np.array(range(lcimin))
        self.CImaxIndex=np.array(((lcimax-1)/(lcimin-1)*np.array(range(lcimin))+0.5)).astype(int)
    else:
        self.CImaxIndex=np.array(range(lcimax))
        self.CIminIndex=np.array(((lcimin-1)/(lcimax-1)*np.array(range(lcimax))+0.5)).astype(int)

    #self.areaComputation()
    #self.computePF()
    #self.PFsmoothing()


def rasterize(self,feature, lat, lon,crs=32641,rfile='raster.tif'):
    from rasterio.io import MemoryFile
    #from affine import Affine
    latdif=np.abs(lat[1:]-lat[:-1])
    londif=np.abs(lon[1:]-lon[:-1])
    Dlat=np.min(latdif[latdif>0])
    Dlon=np.min(londif[londif>0])
    minLat=min(lat)
    minLon=min(lon)
    maxLat=max(lat)
    maxLon=max(lon)
    lenLat=(maxLat-minLat)
    lenLon=(maxLon-minLon)
    nrow=round(lenLat/Dlat+1)
    ncol=round(lenLon/Dlon+1)
    transform = from_origin( maxLon+Dlon/2,maxLat+Dlat/2, Dlon,Dlat)
    with MemoryFile() as memfile:
        meta = {"count": 1, "width": ncol, "height": nrow, "transform": transform, "nodata": float('nan'), "dtype": "float64"}
        with memfile.open(driver='GTiff', **meta) as raster_file:
            raster_file.write(float('nan')*np.ones(ncol*nrow).reshape(1,nrow,ncol))
            self.raster=raster_file.read(1)
            i,j=rasterio.transform.rowcol(raster_file.transform,xs=lon,ys=lat)
            i=i
            j=j
            self.raster[i,j]=feature
            raster_file.write(self.raster, 1)
            raster_file.close()



##########################################BEGIN METHOD tickPositions#################################################

def tickPositions(self,sorted_ranks,Thresholds, HpFA, n = 30):

    TS = np.linspace(Thresholds[0], Thresholds[-1], n)
    unique_rank_positions = HpFA
    P = list([])
    for t in TS:
        ix = np.argmin((Thresholds - t)**2)
        P.append(unique_rank_positions[ix])
    return np.array(P), np.around(TS, 1)

#########################################END METHOD tickPositions####################################################


##########################################BEGIN METHOD __plotTOC#####################################################

def __plotTOC(self,filename = '',title='default',TOCname='TOC',kind='TOC',height=800,width=920,dpi=120,xlabel="default",ylabel="default",autodpi=True,options=np.array(['']),labelsize=7):
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker    #self.tickPositions(self.dThresholds)
    from matplotlib.ticker import MaxNLocator, FuncFormatter, AutoMinorLocator
    #plt.tight_layout()
    if (autodpi==True):
        dpi=int((height+width)*120/(800+920))

    fig, ax1 = plt.subplots(1, 1, figsize=(width/dpi,height/dpi), dpi=dpi)
    title = "TOC/CDF"
    self.unique_ranks = self.Thresholds[self.iunique]
    self.unique_rank_positions = self.HpFA[self.iunique]

    if len(self.unique_ranks) < 30:
        TS = np.linspace(self.Thresholds[0], self.Thresholds[-1], len(self.unique_ranks) )
    else:
        TS = np.linspace(self.Thresholds[0], self.Thresholds[-1], 30)

    P = list([])
    idata=list([])
    for t in TS:
        ix = np.argmin((self.unique_ranks - t)**2)
        P.append(self.unique_rank_positions[ix])
        idata.append(ix)

    self.P = np.array(P)
    self.TS = np.around(TS, 1)

    #Preparing the overlaped plot with the top and right axis for the CDF
    ax2 = fig.add_subplot(111, label = "2", frame_on = False)

    #posiciones de los ticks chiquitos del eje secundario
    P, Q = self.tickPositions(self.rank[self.isorted],self.Thresholds[self.iunique],self.HpFA[self.iunique])



    ax2.set_xticks(P)
    ax2.xaxis.set_minor_locator(ticker.FixedLocator(P))
    tlabel='$X$'
    if (self.featureName!='X'):
        tlabel='$X=$'+self.featureName

    #print(tlabel)
    ax2.set_xlabel(tlabel, color = "tab:blue")
    ax2.set_ylabel("P($x \leq Threshold~~ | ~~y=presence$)", color = "tab:blue")
    ax2.tick_params(labeltop = True)
    ax2.tick_params(labelright = True)
    ax2.tick_params(labelbottom = False)
    ax2.tick_params(labelleft = False)
    ax2.tick_params(labelsize=labelsize)
    ax2.xaxis.tick_top()

    #los ticks grandes
    n = np.argmax(self.P[1:]-self.P[:-1])
    ax2.xaxis.set_major_locator(ticker.FixedLocator(np.unique(np.array(self.P)[[0, n, n+1, -1]])))
    #las etiquetas de los ticks grandes
    ax2.set_xticklabels(np.unique(np.array(self.TS)[[0, n, n+1, -1]]), rotation = 90)
    if ('intersections' in options):
        ax1.vlines(np.array(self.P)[[ n, n+1]],0,self.np,linestyles='dotted',colors="tab:orange",alpha=0.5)
        ax1.hlines(self.Hits[np.array(self.P)[[ n, n+1]]],0,self.ndata,linestyles='dotted',colors="tab:orange",alpha=0.5)

    #parallelogram coordinates
    rx = np.array([0, self.np, self.ndata, self.ndata-self.np, 0])
    ry = np.array([0, self.np, self.np, 0, 0])

    ax1.set_ylim(0, 1.01*self.np)
    ax1.set_xlim(-0.001, 1.01*self.ndata)
    ax1.tick_params(labelsize=labelsize)
    ax1.text(0.575*self.ndata, 0.025*self.np, 'AR = ')
    ax1.text(0.675*self.ndata, 0.025*self.np, str(round(self.areaRatio, 4)))

    #Ploting the uniform distribution line
    ax1.plot(np.array([0, self.ndata]), np.array([0, self.np]),'b-.',
    label = "Random classifier")

    ax1.plot(rx, ry, '--')
    marker='-o'
    markersize=2
    markerb='-o'
    if (kind=='discrete'):
        marker='s'
        markersize=1
        ax1.vlines(self.HpFA[self.iunique],0,
                    self.Hits[self.iunique],colors="tab:red",alpha=0.2)

    #TOC thresholds
    ax1.plot(self.HpFA[self.iunique],
                    self.Hits[self.iunique],
                    marker,markersize = markersize,
                    label = TOCname, linewidth = 1,color = "tab:red")
    if ('boostrapCI' in options):
        if (not self.boostrapFlag):
            self.boostrapCI()
        ax1.plot(self.HpFA[self.CIminIUnique], self.CIminHits[self.CIminIUnique],markerb,markersize = 2*markersize, label = format(self.CImin)+ " CI", linewidth = 2,color = "#311aa3",alpha=0.5)
        ax1.plot(self.HpFA[self.CImaxIUnique], self.CImaxHits[self.CImaxIUnique],markerb,markersize = 2*markersize, label = format(self.CImax)+ " CI", linewidth = 2,color = "#1aa34f",alpha=0.5)
        ax1.fill_between(self.HpFA[self.CIminIUnique], self.CIminHits[self.CIminIUnique], self.CImaxHits[self.CIminIUnique], color='#03e8fc',alpha=0.25)

    ax1.set_xlabel("Hits + False Alarms")
    ax1.set_ylabel("Hits")
    ax1.legend(loc = 'upper left')




    ax2.yaxis.tick_right()
    ax2.yaxis.set_minor_locator(AutoMinorLocator())
    ax2.xaxis.set_label_position('top')
    ax2.yaxis.set_label_position('right')
    ax2.set_title(title,va='baseline')
    if (filename!=''):
        plt.savefig(filename,dpi=dpi)
    else:
        plt.show(block=True)
    plt.close("all")
    plt.clf()
    gc.collect()
    plt.close("all")

##########################################END METHOD __plotTOC#######################################################


##########################################BEGIN METHOD __plotDiscretizedTOC#####################################################

def __plotDiscretizedTOC(self,filename = '',title='default',TOCname='TOC',kind='TOC',height=800,width=920,dpi=120,xlabel="default",ylabel="default",autodpi=True,options=np.array(['']),labelsize=7):
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    from matplotlib.ticker import MaxNLocator, FuncFormatter, AutoMinorLocator

    if (autodpi==True):
        dpi=int((height+width)*120/(800+920))

    fig, ax1 = plt.subplots(1, 1, figsize=(width/dpi,height/dpi), dpi=dpi)

    title = "Discrete (regular) approximation to TOC/CDF"
    self.unique_ranks = self.dThresholds
    self.unique_rank_positions = self.dHpFA

    if len(self.unique_ranks) < 30:
        TS = np.linspace(self.dThresholds[0], self.dThresholds[-1], len(self.unique_ranks) )
    else:
        TS = np.linspace(self.dThresholds[0], self.dThresholds[-1], 30)

    P = list([])
    for t in TS:
        ix = np.argmin((self.unique_ranks - t)**2)
        P.append(self.unique_rank_positions[ix])

    self.P = np.array(P)
    self.TS = np.around(TS, 1)

    #parallelogram coordinates
    rx = np.array([0, self.np, self.ndata, self.ndata-self.np, 0])
    ry = np.array([0, self.np, self.np, 0, 0])

    ax1.set_ylim(0, 1.01*self.np)
    ax1.set_xlim(-0.001, 1.01*self.ndata)
    ax1.tick_params(labelsize=labelsize)
    ax1.text(0.575*self.ndata, 0.025*self.np, 'AUC = ')
    ax1.text(0.675*self.ndata, 0.025*self.np, str(round(self.areaDRatio, 4)))

    #Ploting the uniform distribution line
    ax1.plot(np.array([0, self.ndata]), np.array([0, self.np]),'b-.',
    label = "Random classifier")

    ax1.plot(rx, ry, '--')
    marker='-o'
    markerb='-o'
    markersize=2
    if (self.kind=='discrete'):
        marker='s'
        markersize=1
        ax1.vlines(self.dHpFA,0,
                    self.dHits,colors="tab:red",alpha=0.2)
    #TOC thresholds
    ax1.plot(self.dHpFA,
                    self.dHits,
                    marker,markersize = markersize,
                    label = TOCname, linewidth = 1,color = "tab:red")
    #ax1.plot(self.HpFA[self.iunique],
                    #self.Hits[self.iunique],
                    #'--',markersize = 5.0,
                    #label = TOCname, linewidth = 1)
    if ('boostrapCI' in options):
        if (not self.boostrapFlag):
            self.boostrapCI()
        ax1.plot(self.CIminDHpFA, self.CIminDHits,markerb,markersize = markersize, label = format(self.CImin)+ " CI", linewidth = 2,color = "#311aa3",alpha=0.5)
        ax1.plot(self.CImaxDHpFA, self.CImaxDHits,markerb,markersize = markersize, label = format(self.CImax)+ " CI", linewidth = 2,color = "#1aa34f",alpha=0.5)
        if (len(self.CIminDrank)==len(self.CImaxDrank)):
            ax1.fill_between(self.CIminDHpFA, self.CIminDHits,self.CImaxDHits, color='#03e8fc',alpha=0.15)
            ax1.fill_between(self.CImaxDHpFA,self.CImaxDHits,self.CIminDHits, color='#fbff03',alpha=0.15)
        else:
            ax1.fill_between(self.CIminDHpFA[self.CIminIndex], self.CIminDHits[self.CIminIndex],self.CImaxDHits[self.CImaxIndex], color='#03e8fc',alpha=0.15)
            ax1.fill_between(self.CImaxDHpFA[self.CImaxIndex], self.CImaxDHits[self.CImaxIndex],self.CIminDHits[self.CIminIndex], color='#fbff03',alpha=0.15)

    ax1.set_xlabel("Hits + False Alarms")
    ax1.set_ylabel("Hits")
    ax1.legend(loc = 'upper left')


    #Preparing the overlaped plot with the top and right axis for the CDF
    ax2 = fig.add_subplot(111, label = "2", frame_on = False)

    #posiciones de los ticks chiquitos del eje secundario
    P, Q =self.tickPositions(self.drank,self.dThresholds,self.dHpFA)

    ax2.set_xticks(P)
    ax2.xaxis.set_minor_locator(ticker.FixedLocator(P))

    tlabel='$X$'
    if (self.featureName!='X'):
        tlabel='$X=$'+self.featureName



    ax2.set_xlabel(tlabel, color = "tab:blue")
    ax2.set_ylabel("P($x \leq Threshold~~ | ~~y=presence$)", color = "tab:blue")
    ax2.tick_params(labeltop = True)
    ax2.tick_params(labelright = True)
    ax2.tick_params(labelbottom = False)
    ax2.tick_params(labelleft = False)
    ax2.tick_params(labelsize=labelsize)
    ax2.xaxis.tick_top()

    #los ticks grandes

    n = np.argmax(self.P[1:]-self.P[:-1])

    ax2.xaxis.set_major_locator(ticker.FixedLocator(np.unique(np.array(self.P)[[0, n, n+1, -1]])))

    #las etiquetas de los ticks grandes
    ax2.set_xticklabels(np.unique(np.array(self.TS)[[0, n, n+1, -1]]), rotation = 90)

    ax2.yaxis.tick_right()
    ax2.yaxis.set_minor_locator(AutoMinorLocator())
    ax2.xaxis.set_label_position('top')
    ax2.yaxis.set_label_position('right')
    ax2.set_title(title,va='baseline')
    if (filename!=''):
        plt.savefig(filename,dpi=dpi)
    else:
        plt.show(block=True)
    plt.close("all")
    plt.clf()
    gc.collect()
    plt.close("all")

##########################################END METHOD __plotTOC#######################################################



def __plotCDF(self,filename = '',title='default',TOCname='CDF',kind='CDF',height=800,width=920,dpi=120,xlabel="default",ylabel="default",autodpi=True,options=np.array(['']),labelsize=7):
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    from matplotlib.ticker import MaxNLocator, FuncFormatter, AutoMinorLocator

    if (autodpi==True):
        dpi=int((height+width)*120/(800+920))

    fig, ax1 = plt.subplots(1, 1, figsize=(width/dpi,height/dpi), dpi=dpi)

    title = "Cumulative Distribution Function"
    ax1.set_title(title,va='baseline')
    ax1.set_ylim(0, 1.01)
    ax1.set_xlim(self.minr-0.01*(self.maxr-self.minr),self.maxr)
    ax1.tick_params(labelsize=labelsize)
    #ax1.text(0.575*self.maxr, 0.025, 'AUC = ')
    #ax1.text(0.675*self.maxr, 0.025, str(round(self.areaDRatio, 4)))


    #ax1.plot(rx, ry, '--')
    marker='-o'
    markerb='-o'
    markersize=2
    if (self.kind=='discrete'):
        marker='s'
        markersize=1
    if ('vlines' in options):
        ax1.vlines(self.drank,0,self.CDF,colors="tab:orange",alpha=0.1)
    #TOC thresholds
    ax1.plot(self.drank,
                    self.CDF,
                    marker,markersize = markersize,
                    label = TOCname, linewidth = 1,color = "tab:orange")
    if ('boostrapCI' in options):
        if (not self.boostrapFlag):
            self.boostrapCI()
        ax1.plot(self.CIminDrank, self.CIminCDF,markerb,markersize = markersize, label = format(self.CImin)+ " CI", linewidth = 2,color = "#fc0330",alpha=0.5)
        ax1.plot(self.CImaxDrank, self.CImaxCDF,markerb,markersize = markersize, label = format(self.CImax)+ " CI", linewidth = 2,color = "#fc03c2",alpha=0.5)
        if (len(self.CIminDrank)==len(self.CImaxDrank)):
            ax1.fill_between(self.CIminDrank, self.CIminCDF,self.CImaxCDF, color='#03e8fc',alpha=0.15)
            ax1.fill_between(self.CImaxDrank, self.CImaxCDF,self.CIminCDF, color='#fbff03',alpha=0.15)
        else:
            ax1.fill_between(self.CIminDrank[self.CIminIndex], self.CIminCDF[self.CIminIndex],self.CImaxCDF[self.CImaxIndex], color='#03e8fc',alpha=0.15)
            ax1.fill_between(self.CImaxDrank[self.CImaxIndex], self.CImaxCDF[self.CImaxIndex],self.CIminCDF[self.CIminIndex], color='#fbff03',alpha=0.15)

    #Ploting the uniform distribution line
    ax1.plot(np.array([self.minr, self.maxr]), np.array([0, 1]),'b-.',
    label = "Random classifier")

    tlabel='$Rank$'
    if (self.featureName!='X'):
        tlabel='$Rank=$'+self.featureName

    ax1.set_xlabel(tlabel)
    ax1.set_ylabel("P($x \leq Threshold~~ | ~~y=presence$)")
    ax1.legend(loc = 'lower right')


    if (filename!=''):
        plt.savefig(filename,dpi=dpi)
    else:
        plt.show(block=True)
    plt.close("all")
    plt.clf()
    gc.collect()
    plt.close("all")

##########################################END METHOD __plotTOC#######################################################

def __plotPF(self,filename = '',title='default',TOCname='PF',kind='PF',height=800,width=920,dpi=120,xlabel="default",ylabel="default",autodpi=True,options=np.array(['']),labelsize=7):
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    from matplotlib.ticker import MaxNLocator, FuncFormatter, AutoMinorLocator

    maxpf=np.max(self.PF)
    #print(autodpi)
    if (autodpi==True):
        dpi=int((height+width)*120/(800+920))

    fig, ax1 = plt.subplots(1, 1, figsize=(width/dpi,height/dpi), dpi=dpi)
    if (title=='default'):
        if (kind=='smoothPF'):
            title = "Smoothed Probability Density Function (conditional to presence)"
        else:
            title = "Probability Density Function (conditional to presence)"


    ax1.set_ylim(0, 1.01*np.max(self.PF))
    ax1.set_xlim(self.minr-0.01*(self.maxr-self.minr),self.maxr)
    ax1.tick_params(labelsize=labelsize)
    #ax1.text(0.575*self.maxr, 0.025, 'AUC = ')
    #ax1.text(0.675*self.maxr, 0.025, str(round(self.areaDRatio, 4)))

    #Ploting the uniform distribution line
    ax1.plot(np.array([self.minr, self.maxr]), np.array([1/(self.maxr-self.minr),1/(self.maxr-self.minr)]),'b-.',
    label = "Random classifier")
    #ax1.plot(rx, ry, '--')
    #print(options)
    #print('quartiles' in options)
    if ('vlines' in options):
        if (kind=='smoothPF'):
            ax1.vlines(self.drank,self.smoothPF,maxpf,colors="tab:gray",alpha=0.5,linewidth = 0.2)
            if ('quartiles' in options):
                ax1.vlines(self.drank,0,self.smoothPF,colors="#2c03fc",alpha=0.05,linewidth = 0.1)
            else:
                ax1.vlines(self.drank,0,self.smoothPF,colors="#2c03fc",alpha=0.5,linewidth = 0.2)
        else:
            ax1.vlines(self.drank,self.PF,maxpf,colors="tab:gray",alpha=0.5,linewidth = 0.2)
            if ('quartiles' in options):
                ax1.vlines(self.drank,0,self.PF,colors="#2c03fc",alpha=0.05,linewidth = 0.1)
            else:
                ax1.vlines(self.drank,0,self.PF,colors="#2c03fc",alpha=0.5,linewidth = 0.2)


    if ('quartiles' in options):
        i1=np.argmax(self.CDF>0.25)
        i2=np.argmax(self.CDF>0.5)
        i3=np.argmax(self.CDF>0.75)
        ax1.fill_between(self.drank[0:i1+1],self.PF[0:i1+1],color='#fc9d03',alpha=0.55)
        ax1.fill_between(self.drank[i1:i2+1],self.PF[i1:i2+1],color='#fc0703',alpha=0.55)
        ax1.fill_between(self.drank[i2:i3+1],self.PF[i2:i3+1],color='#ca03fc',alpha=0.55)
        ax1.fill_between(self.drank[i3:],self.PF[i3:],color='#016e32',alpha=0.55) ##fc3d03

    marker='-o'
    markersize=0.5
    if (self.kind=='discrete'):
        marker='h'
        markersize=3
        if (kind=='smoothPF'):
            ax1.vlines(self.drank,0,self.smoothPF,colors="#4287f5",alpha=0.95,linewidth = 3)
        else:
            ax1.vlines(self.drank,0,self.PF,colors="#4287f5",alpha=0.95,linewidth = 3)
        if (title=='default'):
            title="Mass Probability Function (conditional to presence)"
            if (kind=='smoothPF'):
                title="Regularized Mass Probability Function (conditional to presence)"

    ax1.set_title(title,va='baseline')
    #TOC thresholds
    if (kind=='smoothPF'):
        ax1.plot(self.drank,self.smoothPF,marker,markersize = markersize,label = 'Smoothed PF', linewidth = 1,color = "#4287f5")
        #ax1.plot(self.drank,self.PF,marker,markersize = markersize/4,label = 'Original PF', linewidth = 0.4,color = "#fa4807" )
    else:
        ax1.plot(self.drank, self.PF,marker,markersize = markersize,label = TOCname, linewidth = 1,color = "#4287f5")


    if ('boostrapCI' in options):
        if (not self.boostrapFlag):
            self.boostrapCI()
        if (kind=='smoothPF'):
            ax1.plot(self.CIminDrank, self.CIminSmoothPF,marker,markersize = markersize, label = format(self.CImin)+ " CI", linewidth = 2,color = "#fc1303",alpha=0.5)
            ax1.plot(self.CImaxDrank, self.CImaxSmoothPF,marker,markersize = markersize, label = format(self.CImax)+ " CI", linewidth = 2,color = "#fc9003",alpha=0.5)
            if (len(self.CIminDrank)==len(self.CImaxDrank)):
                ax1.fill_between(self.CIminDrank, self.CIminSmoothPF,self.CImaxSmoothPF, color='#03e8fc',alpha=0.15)
                ax1.fill_between(self.CImaxDrank, self.CImaxSmoothPF,self.CIminSmoothPF, color='#fbff03',alpha=0.15)
            else:
                ax1.fill_between(self.CIminDrank[self.CIminIndex], self.CIminSmoothPF[self.CIminIndex],self.CImaxSmoothPF[self.CImaxIndex], color='#03e8fc',alpha=0.15)
                ax1.fill_between(self.CImaxDrank[self.CImaxIndex], self.CImaxSmoothPF[self.CImaxIndex],self.CIminSmoothPF[self.CIminIndex], color='#fbff03',alpha=0.15)
        else:
            ax1.plot(self.CIminDrank, self.CIminPF,marker,markersize = markersize, label = format(self.CImin)+ " CI", linewidth = 2,color = "#03e8fc",alpha=0.5)
            ax1.plot(self.CImaxDrank, self.CImaxPF,marker,markersize = markersize, label = format(self.CImax)+ " CI", linewidth = 2,color = "#fc9003",alpha=0.5)
            if (len(self.CIminDrank)==len(self.CImaxDrank)):
                ax1.fill_between(self.CIminDrank, self.CIminPF,self.CImaxPF, color='#03e8fc',alpha=0.15)
                ax1.fill_between(self.CImaxDrank, self.CImaxPF,self.CIminPF, color='#fbff03',alpha=0.15)
            else:
                ax1.fill_between(self.CIminDrank[self.CIminIndex], self.CIminPF[self.CIminIndex],self.CImaxPF[self.CImaxIndex], color='#03e8fc',alpha=0.25)
                ax1.fill_between(self.CImaxDrank[self.CImaxIndex], self.CImaxPF[self.CImaxIndex],self.CIminPF[self.CIminIndex], color='#fbff03',alpha=0.25)

    tlabel='$Rank$'
    if (self.featureName!='X'):
        tlabel='$Rank=$'+self.featureName

    ax1.set_xlabel(tlabel)
    ax1.set_ylabel("P($x = Threshold~~ | ~~y=presence$)")
    ax1.legend(loc = 'center right')


    if (filename!=''):
        plt.savefig(filename,dpi=dpi)
    else:
        plt.show(block=True)
    plt.close("all")
    plt.clf()
    gc.collect()
    plt.close("all")



def __plotDPF(self,filename = '',title='default',TOCname='CDF',kind='CDF',height=800,width=920,dpi=120,xlabel="default",ylabel="default",autodpi=True,options=np.array(['']),labelsize=7):
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    from matplotlib.ticker import MaxNLocator, FuncFormatter, AutoMinorLocator

    maxDPF=np.max(self.DPF)
    minDPF=np.min(self.DPF)
    maxSmoothDPF=np.max(self.smoothDPF)
    minSmoothDPF=np.min(self.smoothDPF)
    if (autodpi==True):
        dpi=int((height+width)*120/(800+920))

    fig, ax1 = plt.subplots(1, 1, figsize=(width/dpi,height/dpi), dpi=dpi)
    if(title=='default'):
        if (kind=='smoothDPF'):
            title = "Smoothed First Derivative of the Probability Density Function"
        else:
            title = "First Derivative of the Probability Density Function"
    if (kind=='smoothDPF'):
        ax1.set_ylim(1.01*minSmoothDPF, 1.01*maxSmoothDPF)
    else:
        ax1.set_ylim(1.01*minDPF, 1.01*maxDPF)
    ax1.set_xlim(self.minr-0.01*(self.maxr-self.minr),self.maxr)
    ax1.tick_params(labelsize=labelsize)

    #Ploting the uniform distribution line
    ax1.plot(np.array([self.minr, self.maxr]), np.array([0,0]),'b-.',
    label = "Random classifier")

    if ('vlines' in options):
        if (kind=='smoothDPF'):
            ax1.vlines(self.drank[self.smoothDPF>0],self.smoothDPF[self.smoothDPF>0],maxSmoothDPF,colors="tab:gray",alpha=0.45,linewidth = 0.3)
            ax1.vlines(self.drank[self.smoothDPF<0],self.smoothDPF[self.smoothDPF<0],minSmoothDPF,colors="tab:gray",alpha=0.45,linewidth = 0.3)
            ax1.vlines(self.drank[self.smoothDPF==0],maxSmoothDPF,minSmoothDPF,colors="tab:gray",alpha=0.45,linewidth = 0.3)
        else:
            ax1.vlines(self.drank[self.DPF>0],self.DPF[self.DPF>0],maxDPF,colors="tab:gray",alpha=0.45,linewidth = 0.3)
            ax1.vlines(self.drank[self.DPF<0],self.DPF[self.DPF<0],minDPF,colors="tab:gray",alpha=0.45,linewidth = 0.3)
            ax1.vlines(self.drank[self.DPF==0],maxDPF,minDPF,colors="tab:gray",alpha=0.45,linewidth = 0.3)


    marker='-o'
    markersize=1
    fmt=''
    if (self.kind=='discrete'):
        marker='s'
        fmt='s'
        markersize=3
        if (kind=='smoothDPF'):
            ax1.vlines(self.drank,0,self.smoothDPF,colors="#4287f5",alpha=0.95,linewidth = 1)
        else:
            ax1.vlines(self.drank,0,self.DPF,colors="#4287f5",alpha=0.95,linewidth = 1)
        if(title=='default'):
                title="First Difference of the Mass Probability Function"
                if (kind=='smoothDPF'):
                    title="Regularized Difference of the Mass Probability Function"

    ax1.set_title(title,va='baseline')

    #TOC thresholds
    if (kind=='smoothDPF'):
        ##print('marker',marker)
        #ax1.plot(self.drank,self.DPF,  marker,markersize = markersize-1,label = 'Original DPF', linewidth = 0.25,color = "#b7bec9")
        #ax1.plot(self.drank,self.smoothDPF,  marker,markersize = markersize,label = 'Smoothed DPF', linewidth = 1,color = "#4287f5")
        #ax1.plot(self.drank,self.DPF,marker='s',markersize =0,label = 'Original DPF', linewidth = 0.25,color = "#fa4807" ,alpha=0.5)
        ax1.plot(self.drank,self.smoothDPF,marker='s',markersize = 0.5,label = 'Smoothed DPF', linewidth = 1,color = "#4287f5",alpha=1)
    else:
        ax1.plot(self.drank,self.DPF,marker,markersize = markersize,label = TOCname, linewidth = 1,color = "#4287f5")

    if ('quartiles' in options):
        i1=np.argmax(self.CDF>0.25)
        i2=np.argmax(self.CDF>0.5)
        i3=np.argmax(self.CDF>0.75)
        ax1.fill_between(self.drank[0:i1+1],self.DPF[0:i1+1],color='#fc9d03',alpha=0.5)
        ax1.fill_between(self.drank[i1:i2+1],self.DPF[i1:i2+1],color='#fc0703',alpha=0.5)
        ax1.fill_between(self.drank[i2:i3+1],self.DPF[i2:i3+1],color='#ca03fc',alpha=0.5)
        ax1.fill_between(self.drank[i3:],self.DPF[i3:],color='#016e32',alpha=0.5)

    if ('boostrapCI' in options):
        if (not self.boostrapFlag):
            self.boostrapCI()
        if (kind=='smoothDPF'):
            ax1.plot(self.CIminDrank, self.CIminSmoothDPF,marker,markersize = markersize, label = format(self.CImin)+ " CI", linewidth = 2,color = "#4e03fc",alpha=0.5)
            ax1.plot(self.CImaxDrank, self.CImaxSmoothDPF,marker,markersize = markersize, label = format(self.CImax)+ " CI", linewidth = 2,color = "#03b6fc",alpha=0.5)
            if (len(self.CIminDrank)==len(self.CImaxDrank)):
                ax1.fill_between(self.CIminDrank, self.CIminSmoothDPF,self.CImaxSmoothDPF, color='#03e8fc',alpha=0.15)
                ax1.fill_between(self.CImaxDrank, self.CImaxSmoothDPF,self.CIminSmoothDPF, color='#fbff03',alpha=0.15)
            else:
                ax1.fill_between(self.CIminDrank[self.CIminIndex], self.CIminSmoothDPF[self.CIminIndex],self.CImaxSmoothDPF[self.CImaxIndex], color='#03e8fc',alpha=0.15)
                ax1.fill_between(self.CImaxDrank[self.CImaxIndex], self.CImaxSmoothDPF[self.CImaxIndex],self.CIminSmoothDPF[self.CIminIndex], color='#fbff03',alpha=0.15)
        else:
            ax1.plot(self.CIminDrank, self.CIminDPF,marker,markersize = markersize, label = format(self.CImin)+ " CI", linewidth = 2,color = "#4e03fc",alpha=0.5)
            ax1.plot(self.CImaxDrank, self.CImaxDPF,marker,markersize = markersize, label = format(self.CImax)+ " CI", linewidth = 2,color = "#fbff03",alpha=0.5)
            if (len(self.CIminDrank)==len(self.CImaxDrank)):
                ax1.fill_between(self.CIminDrank, self.CIminDPF,self.CImaxDPF, color='#03e8fc',alpha=0.25)
                ax1.fill_between(self.CImaxDrank, self.CImaxDPF,self.CIminDPF, color='#fbff03',alpha=0.25)
            else:
                ax1.fill_between(self.CIminDrank[self.CIminIndex], self.CIminDPF[self.CIminIndex],self.CImaxDPF[self.CImaxIndex], color='#03e8fc',alpha=0.15)
                ax1.fill_between(self.CImaxDrank[self.CImaxIndex], self.CImaxDPF[self.CImaxIndex],self.CIminDPF[self.CIminIndex], color='#fbff03',alpha=0.15)

    tlabel='$Rank$'
    if (self.featureName!='X'):
        tlabel='$Rank=$'+self.featureName

    ax1.set_xlabel(tlabel)
    ax1.set_ylabel(r'$\frac{D}{Dx}~P(x = Threshold~~ | ~~y=presence$)')
    ax1.legend(loc = 'center right')


    if (filename!=''):
        plt.savefig(filename,dpi=dpi)
    else:
        plt.show(block=True)
    plt.close("all")
    plt.clf()
    gc.collect()
    plt.close("all")




def __plotRaster(self,filename = '',title='default',TOCname='Raster',kind='raster',height=800,width=920,dpi=120,xlabel="default",ylabel="default",autodpi=True,options=np.array(['']),labelsize=7):
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    from matplotlib.ticker import MaxNLocator, FuncFormatter, AutoMinorLocator


    cmap = mpl.colormaps['RdBu']  # viridis is the default colormap for imshow
    if ('binary' in options):
        #cmap = mpl.colors.ListedColormap(['#67001f', '#053061'])
        cmap = mpl.colors.ListedColormap(['red', '#053061'])

    cmap.set_bad(color='gray')
    plot=plt.imshow(self.raster, cmap=cmap)
    plt.colorbar()
    plt.minorticks_on()
    tlabel=TOCname
    if (TOCname=='Raster'):
        tlabel='$Rank=$'+self.featureName

    plt.title(tlabel,loc='center')

    if (filename!=''):
        plt.savefig(filename,dpi=dpi)
    else:
        plt.show(block=True)
    plt.close("all")
    plt.clf()
    gc.collect()
    plt.close("all")


def __plotHist(self,filename = '',title='default',TOCname='Raster',kind='raster',height=800,width=920,dpi=120,xlabel="default",ylabel="default",autodpi=True,options=np.array(['']),labelsize=7):
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    from matplotlib.ticker import MaxNLocator, FuncFormatter, AutoMinorLocator
    if (not self.boostrapFlag):
        self.boostrapCI()

    height, bins, patches =plt.hist(self.boostrapAreas,15,color='#9c0505')
    plt.title("Histogram of bootstrap area ratios of the TOC")
    plt.fill_betweenx([0, height.max()], self.CIminArea, self.CImaxArea, color='#ff0303', alpha=0.1)

    if (filename!=''):
        plt.savefig(filename,dpi=dpi)
    else:
        plt.show(block=True)
    plt.close("all")
    plt.clf()
    gc.collect()
    plt.close("all")





##########################################BEGIN METHOD plot#####################################################

#This function plots the TOC to the terminal or to a file
def plot(self,filename = '',title='default',TOCname='default',kind='None',height=800,width=920,dpi=120,xlabel="default",ylabel="default",autodpi=True,options=np.array(['']),labelsize=7):
    """
    A generic plot function for all the kind of TOCs.  All the parameters are optional. If ``filename`` is not given it plots to a window, otherwise it is a png file.

    :param filename: Optional. If given it must be a png filename, otherwise the TOC is plotted to a window.

    :param title: Optional, title of the plot.

    :param kind: Optional, a standard TOC can be plotted normalized or in the original axis values.

    :param height: pixels of the height. 1800 by default.

    :param width: pixels of the width. 1800 by default.

    :param dpi: resolution. 300 by default.

    :param xlabel: string.

    :param ylabel: string.

    :return: it does not return anything.

    """

    if (kind=='None'):
        kind=self.kind
    if (kind=='continuous' or kind=='semicontinuous' or kind=='forcedContinuous' or kind=='discrete'):
        if (TOCname=='default'):
            TOCname=kind+" TOC"
        self.__plotTOC(filename,title,TOCname,kind,height,width,dpi,xlabel,ylabel,autodpi,options,labelsize)
    if (kind=='discretization'):
        if (TOCname=='default'):
            TOCname="Discrete approx. of the TOC"
        self.__plotDiscretizedTOC(filename,title,TOCname,kind,height,width,dpi,xlabel,ylabel,autodpi,options,labelsize)
    if (kind=='CDF'):
        if (TOCname=='default'):
            TOCname="Cumulatve Distribution Function"
        self.__plotCDF(filename,title,TOCname,kind,height,width,dpi,xlabel,ylabel,autodpi,options,labelsize)
    if (kind=='PF' or kind=='smoothPF'):
        if (TOCname=='default'):
            TOCname=kind
        self.__plotPF(filename,title,TOCname,kind,height,width,dpi,xlabel,ylabel,autodpi,options,labelsize)
    if (kind=='DPF' or kind=='smoothDPF'):
        if (TOCname=='default'):
            TOCname=kind
        self.__plotDPF(filename,title,TOCname,kind,height,width,dpi,xlabel,ylabel,autodpi,options,labelsize)
    if (kind=='raster'):
        if (TOCname=='default'):
            TOCname='Raster'
        self.__plotRaster(filename,title,TOCname,kind,height,width,dpi,xlabel,ylabel,autodpi,options,labelsize)
    if (kind=='histogram'):
        if (TOCname=='default'):
            TOCname='Histogram'
        self.__plotHist(filename,title,TOCname,kind,height,width,dpi,xlabel,ylabel,autodpi,options,labelsize)


##########################################END METHOD plot#######################################################



TOCPF.discretization=discretization
TOCPF.continuity=continuity
TOCPF.areaComputation=areaComputation
TOCPF.tickPositions=tickPositions
TOCPF.areaDComputation=areaDComputation
TOCPF.computePF=computePF
TOCPF.centeredDF=centeredDF
TOCPF.PFsmoothing=PFsmoothing
TOCPF.RLS=RLS
TOCPF.fitNN=fitNN
TOCPF.meanWindowSmoothing=meanWindowSmoothing
TOCPF.integrateTrapezoidal=integrateTrapezoidal
TOCPF.rank2prob=rank2prob
TOCPF.areaComputationBoostrap=areaComputationBoostrap
TOCPF.boostrapCI=boostrapCI
TOCPF.boostrapContinuity=boostrapContinuity
TOCPF.boostrapComputePF=boostrapComputePF
TOCPF.boostrapPFsmoothing=boostrapPFsmoothing
TOCPF.rasterize=rasterize
TOCPF.simulate=simulate
TOCPF.__plotTOC=__plotTOC
TOCPF.__plotCDF=__plotCDF
TOCPF.__plotPF=__plotPF
TOCPF.__plotDPF=__plotDPF
TOCPF.__plotDiscretizedTOC=__plotDiscretizedTOC
TOCPF.__plotRaster=__plotRaster
TOCPF.__plotHist=__plotHist
TOCPF.plot=plot
