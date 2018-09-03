"""
# -----------------------------------------------------------------------------
# Created: on 13-Aug-2018, updated 15-Aug-2018
# Author: Suksmono@{STEI-ITB, MDR Inc.}
# Summary: By reperesenting binary vectors as node, we search M-orthoset or
#          M-order clique by growing a random graph
# -----------------------------------------------------------------------------
#    ALGORITHM
# -----------------------------------------------------------------------------
#(0) DEFINE 
#        -Problem size: H-Order M
#        -Problem effort: MAXITR, tolerance: eps, NSWEEP, NREADS
#(1). INITialization (from the start or continues from file)
#        -DISCRETE STRUCTURES
#            -Master graph: G<-{}
#            -Record of trial TRIAL<- {}
#            -List of cliques: NLS_CLQ<- Clique(G), ordered by |tCLQ|
#                -Init master-Clique: CLQ <-MaxClique(tCLQ)
#        -PARAMS: NV<-|CLQ|, itr<-0
#(2) WHILE (NV< H-ORDER and itr<MAXITR):   
#    -From Largest to Smallest, fetch tCLQ in NLS_CLQ not yet in TRIAL
#        -Convert into matrix of variables: tCLQ->sk
#        -Get next ortho vectors: 
#            -update list of tried tCLQ: TRIAL<-TRIAL U tCLQ
#            -get vSol
#        - ----------------- replaced ------------------
#        -#Generate complete graph from F<-{vSol} U tCLQ
#        -#Combine F to master: G<-F U G
#        - ----------------- --------- ------------------
#        -Insert nodes into graph: G<-{vSol}
#        -Create adjacency matrix: A
#        -Update graph structure based on adjcy mtx: G<-fUpd(A)
#        - ----------------- not yet implemented ------------------
#        #Remove low -degree nodes from grap: G<- G-badNodes(G) 
#        #Remove list in trial incl bad nodes: TRIAL<-TRIAL-[badNodes(G)]
#        - ----------------- --------- ------------------
#    -Update List of Clique: LS_CLQ=[{tCLQ}]<-Clique(G)
#        -Update master clique: CLQ <- Max(tCLQ)
#    ## save to file
#    -save structures into file    
#    -itr=itr+1, NV=|CLQ|
# -----------------------------------------------------------------------------
"""

# -----------------------------------------------------------
# IMPORT PACKAGES AND MODULES
# -----------------------------------------------------------
#from sympy import *
import numpy as np
import networkx as nx
#from networkx.algorithms.approximation import clique as xclq
import matplotlib.pyplot as plt
import pickle as pk
import multiprocessing as mp
from multiprocessing import Pool
#from os import getpid
import neal
import pandas as pd
#import sys
#
# ****************************************************************************
# import from prb_iof
# readHMatrix, writeHMatrix
# ****************************************************************************
def writeHMatrix(fname, matx):
   # df=pd.DataFrame(data=matx.astype(int))
    df=pd.DataFrame(data=matx)
    df.to_csv(fname,sep=',', header=False, index=False)
    
def readHMatrix(fname):
    data=pd.read_csv(fname,sep=',', header=None )
    return data.as_matrix() #.tolist()

# ****************************************************************************
# import from prb_graph
# from prb_graph import svec2hex, hex2svec, CLQ2svlist, sortCLQS, isOrthoSet,\
#                      maxCLQ, listSCLQ, matxIndicator
# ****************************************************************************

#------------------------------------------------------------
#numerical: q-to-s and s-to-q transforms
#------------------------------------------------------------
 # define function vq2s
def vq2s(x):
    return(1-2*x)
# define function vs2q
def vs2q(x):
    return(1/2-x/2)

#-----------------------------------------
# s-vector -> hexa 
#-----------------------------------------
def svec2hex(v):
    tVal=0
    NC=len(v)
    for m in range(0,NC):
        tV=v[m]
        tQ=int(vs2q(tV))
        tVal=tVal+tQ*2**(NC-m-1)
        #print(tQ,tVal)
#    return hex(int(tVal)), int(tVal)
    return hex(int(tVal))

#-----------------------------------------
# Hexa -> s-vector
#-----------------------------------------
def hex2svec(v,N):
    #bchar=bin(int(v, 16))[2:]
    bchar=format(int(v, 16), 'b').zfill(N)
    NC=len(bchar)
    tVec=[]
    for m in range(0,NC):
        tV=int(bchar[m])
        tV1=int(vq2s(tV))
        tVec.append(tV1)
        #print(tVec)
    return tVec

#-----------------------------------------
#        sk=CLQ2sk(CLQ)
#-----------------------------------------
def CLQ2svlist(CLQ,M):
    sk=[np.int_(np.ones(M)).tolist()]
    #while nn in range(0,len(CLQ)):
    listCLQ=list(CLQ)
    #while len(ttCLQ)>0:
    for mm in range(0,len(listCLQ)):
        #tndHex=ttCLQ.pop()
        tndHex=listCLQ[mm]
        sk.append(hex2svec(tndHex,M))
    return(sk)

#-----------------------------------------
#        sortCLQS
#-----------------------------------------
def sortCLQS(allCLQ):
    # find out order of all cliques in graph
    # input : all cliques [ [1,3], [2,4,5,7], [0] ]
    # output: sorted all cliques [ [2,4,5,7], [1,3], [0] ] and set
    #        [ (2,4,5,7), (1,3), (0) ]

    tLQ=np.shape(allCLQ)
    LQ=tLQ[0]
    vLQ=list()
    for m in range(0,LQ):
        vLQ.append(len(allCLQ[:][m]))
    #print('Cliques orders in graph:\n',vLQ)
    
    # get sorted index
    clqIdx=np.argsort(vLQ)
    # arrange allCLQ according to length
    #print('Descending sorted cliques')
    NQ=len(vLQ)
    sCLQ=[]
    #setCLQ=[]
    for m in range(0,NQ):
        tIdx=NQ-m-1
        #print(tIdx)
        sCLQ.append(allCLQ[:][clqIdx[tIdx]])
        #setCLQ.append(set(allCLQ[:][clqIdx[tIdx]]))
        #print(sCLQ)
    #
    return sCLQ #, setCLQ

#-----------------------------------------
#        isOrthoSet
#-----------------------------------------
def isOrthoSet(CLQ,M):
    sk=CLQ2svlist(CLQ,M)
    D=np.abs(np.matmul(sk, np.transpose(sk)))  
    N=len(sk)
    TF= (np.sum(D)-M*N) == 0 
    return TF #, D
#-----------------------------------------
#        matxIndicator
#-----------------------------------------
def matxIndicator(CLQ,M):
    sk=CLQ2svlist(CLQ,M)
    D=np.abs(np.matmul(sk, np.transpose(sk)))  
    return D
#-----------------------------------------
#       maxCLQ
#-----------------------------------------
def maxCLQ(sCLQ):
    if len(sCLQ)<1:
        return set()
    else:
        vIdx=list()
        for m in range(0,len(sCLQ)):
            vIdx.append(len(sCLQ[:][m]))
        # get max 
        idxMax=np.argmax(vIdx)
        myClique=set(sCLQ[:][idxMax])
        return myClique

#-----------------------------------------
#       listCLQ
#-----------------------------------------
def listSCLQ(sCLQ):
    if len(sCLQ)<1:
        return len(set())
    else:
        vIdx=list()
        for m in range(0,len(sCLQ)):
            vIdx.append(len(sCLQ[:][m]))
    return vIdx
                      
# ****************************************************************************
           

#-----------------------------------------
def IsingCoeffsNEW(sk):
    [NC, M]=np.shape(sk)
    hi=np.zeros(M)
    Jij=np.zeros([M,M])
    A=np.zeros([M,M])
    hi=[]
    for m in range(0,len(sk)):
        vi=sk[:][m]
        A=A+2*np.outer(vi, vi)
   
    #
    b=len(sk)*M
    # -- print result --
    #print(A)
    for m in range(0,M):
        hi.append(0. )
        for nn in range(m+1, M):
          Jij[m][nn]=A[m][nn]    
    # 
    return b, hi, Jij
#-----------------------------------------

def findOrthoVec_PAR(vParams):
    ''' extract parameters '''
    sk=vParams[:][0]
    #lsk=len(sk)
    NSWEEPS=vParams[:][1]
    NREADS=vParams[:][2]
    
    '''EXTRACT ISING PARAMETERS  '''

    b, hi, Jij = IsingCoeffsNEW(sk)

    # normalize coefficients PROBABLY UNNECESSARY
    aJij=np.abs(Jij)
    ahi=np.abs(hi)
    maxCoeff=np.max([np.max(ahi), np.max(aJij)])
    hi=hi/maxCoeff
    Jij=Jij/maxCoeff
    #
    b=b/maxCoeff
    
    '''
    -----------------------------------------------------------------------------
    convert the problem into Ising coefficients
    -----------------------------------------------------------------------------
    '''
    #in dictionary format
    h={0:0}
    J={(0,1):1}
    
    for m in range(0,len(hi)):
        h[m]=hi[m]
        for n in range (m+1,len(hi)):
            J[m,n]=Jij[m,n]
        
    '''
    -----------------------------------------------------------------------------
    # SOLVE THE PROBLEM
    -----------------------------------------------------------------------------
    select a solver
    > dimod: ExaxtSolver
    > neal:  SimulatedAnnealingSampler
    '''
    #
    print('Solving the problem using neal  ...')
    solver=neal.SimulatedAnnealingSampler()
    #NSWEEPS=1*1*10*10*1000
    response = solver.sample_ising(h, J, sweeps=NSWEEPS, num_reads=NREADS)
    #
    vE=response.data_vectors['energy']
    aSol=response.samples_matrix
    #print('All energy',vE)
    #print('All configurations',aSol)
    # report all minimum energy
    minE=min(vE)
    idxVE=[i for i, e in enumerate(vE) if e == minE]
    lSol=aSol.tolist()
    vmSol=list()
    # report all ortho-vectors 
    for mm in range(0,len(idxVE)):
        tVect=lSol[:][mm]
        if (not( (tVect in vmSol) or (np.negative(tVect).tolist() in vmSol) )):
            vmSol.append(tVect)        
    #################################  
    #return b, minE, vmSol
    gapE=abs(b+minE)
    t=list()
    #t.append(b)
    #t.append(minE)
    t.append(gapE)
    t.append(vmSol)
    return t
    #################################   
 
#------------------------------------
# define functions
#------------------------------------
def updGStruct(G):
    # list all nodes into set
    ndsG=set(G.nodes())
    # construct a list of H-vectors
    vH=[]
    vHex=list()
    while (len(ndsG)>0):
        tHex=ndsG.pop()
        vHex.append(tHex)
        vH.append(hex2svec(tHex,M))        
    #create adjacency matrix
    # D=np.abs(np.matmul(sk, np.transpose(sk)))  
    A=np.abs(np.matmul(vH, np.transpose(vH)))
    G=nx.Graph()
    for m in range(0,len(vH)):
        for nn in range(m+1,len(vH)):
            if A[m][nn]==0:
                G.add_edge(vHex[m], vHex[nn])
            #--end-if
        #--end for nn
    #-- end for m
    return(G)
####
def genSHVector(M):
    rd=np.random.permutation(M)
    v=np.ones(M).tolist()
    for mm in range(0,int(M/2)):
        v[rd[mm]]=-1.
    #
    return(v)

#----------------------------------------------------------------------------
# Addendum: construct input of parallel jobs
#----------------------------------------------------------------------------
#----------------------------------------------------------------------------
# generate job list
# description: create a list of job based on the graph G and TRIAL 
# input: G, TRIAL
# output: LST_JOB : it shoudl be a kind of database of level G
#       -[ [] ]
#       - orderered by length of tClique to extend
#       - clean from alredy tried jobs, wether succesful or not
#----------------------------------------------------------------------------
def createJobList(G,TRIAL):
    # create list of clique from G: non-ordered list of flique
    tLS_CLQ=list(nx.clique.find_cliques(G))
    # clean-up from TRIAL
    #tLS_CLQ=NLS_CLQ.copy()
    ttLS_CLQ=[]
    for m in range(0, len(tLS_CLQ)):
        tls_clq=tuple(np.sort(tLS_CLQ[m]))
        if not(tls_clq in TRIAL):
            ttLS_CLQ.append(tls_clq)
        #--end if not
    #-- end for
    return(ttLS_CLQ)

#----------------------------------------------------------------------------
# get NJOBS from JOBLIST, format into standard input of
# multiprocessing vParams
#  vParams=[ \
#         [ sk, NTSWEEPS, NREADS] \
#         ]
#----------------------------------------------------------------------------

def requestJobs(JOB_LIST, M, NJOBS, NTSWEEPS, NREADS):
    m=0
    # --add
    tJOB_LIST=JOB_LIST.copy()
    #TF=True
    vParams1=[]
    NDZ_svec=np.int_(np.ones(M)).tolist()
    #NDZ_hex=svec2hex(NDZ_svec)
    while(m<len(JOB_LIST) and m<NJOBS):
        tCLQ=JOB_LIST[m]
        # construct tsk
        tsk=[]
        tsk.append(NDZ_svec)
        for nn in range(0, len(tCLQ)):
            tndsvec=hex2svec(tCLQ[nn],M)
            tsk.append(tndsvec)
        # end for
        tPar=[tsk, NTSWEEPS, NREADS]
        vParams1.append(tPar)
        # --add
        tJOB_LIST.remove(tCLQ)
        m=m+1
    #
    return vParams1, tJOB_LIST
    
#----------------------------------------------------------------------------
# update TRIAL
# input: TRIAL, vParams
# output: updated TRIAL
#----------------------------------------------------------------------------
def updateTRIAL(TRIAL, vParams):
    for m in range(0,len(vParams)):
        tvpar=vParams[:][m]
        tlst_clq=tvpar[:][0]
        # convert list of svec to tuple of ordered hex
        # exclude first entyry
        NN, MM=np.shape(tlst_clq)
        tls_clqHex=list()
        for nn in range(1, NN):
            tHex=svec2hex(tlst_clq[nn])
            tls_clqHex.append(tHex)
        # -- end for
        tTpl=tuple(np.sort(tls_clqHex))
        TRIAL.add(tTpl)
    #-- end for
    return(TRIAL)

#----------------------------------------------------------------------------
# update Graph and Clique after finding new ortho-vectors
#----------------------------------------------------------------------------
def updateGRAPH(tV1, G, CLQ, MAXBRANCH):
    #
    for kkk in range(0, len(tV1)):
        sol0=tV1[:][kkk]
        gapE=sol0[0]
        vmSol=sol0[1]
        print('Sol #', kkk, '-> Energy Gap:', gapE)
        vgapE.append(gapE)    
        # if orthogonal, add to orthoset sk
        if ( (gapE/M)<eps):
            #print('CLQ(G) before:', len(xclq.max_clique(G)), ', |G|=', len(G.nodes()))
            NSOL=min(MAXBRANCH,len(vmSol))
            for mm in range(0,NSOL):
                tndHex=svec2hex(vmSol[:][mm])
                G.add_node(tndHex)
                
            #--end for mm ---
            # update graph structure in G
            G=updGStruct(G)
            tndHex=svec2hex(vmSol[:][0])
            CLQ.add(tndHex)
            #
    return(G,CLQ)
####
    #def updateGRAPH(tV1, G, CLQ, MAXBRANCH):
def updateGRAPH_MOD(tV1, G, CLQ, MAXBRANCH, MAXSOL):
    #cntV=0
    tMaxSol=0
    kkk=0
    #for kkk in range(0, len(tV1)):
    while kkk <len(tV1) and tMaxSol<MAXSOL:
        sol0=tV1[:][kkk]
        gapE=sol0[0]
        vmSol=sol0[1]
        print('Sol #', kkk, '-> Energy Gap:', gapE)
        vgapE.append(gapE)    
        # if orthogonal, add to orthoset sk
        if ( (gapE/M)<eps):
            #print('CLQ(G) before:', len(xclq.max_clique(G)), ', |G|=', len(G.nodes()))
            NSOL=min(MAXBRANCH,len(vmSol))
            for mm in range(0,NSOL):
                tndHex=svec2hex(vmSol[:][mm])
                G.add_node(tndHex)
                
            #--end for mm ---
            # update graph structure in G
            G=updGStruct(G)
            tndHex=svec2hex(vmSol[:][0])
            CLQ.add(tndHex)
            #
            tMaxSol=tMaxSol+1
        kkk=kkk+1
    return(G,CLQ)
#----------------------------------------------------------------------------
# list all clique in graph
# *****************************************************************************
#----------------------------------------------------------------------------
# calculate average degree of a graph 
#----------------------------------------------------------------------------
def mean_degree(G):
    lstNodes=list(G.nodes())
    NNodes=len(lstNodes)
    cntDeg=0
    for m in range(0,NNodes):
        cntDeg=cntDeg+G.degree(lstNodes[m])
    #
    return(cntDeg/NNodes)

#----------------------------------------------------------------------------
# show distribution of degree of a graph 
#----------------------------------------------------------------------------
def hist_degree(G, num_bins):
    # show distribution of degree in G
    lstNodes=list(G.nodes())
    NNodes=len(lstNodes)
    lstDeg=list()
    for m in range(0,NNodes):
        lstDeg.append(G.degree(lstNodes[m]))
    n, bins, patches = plt.hist(lstDeg, num_bins, facecolor='blue', alpha=0.5)
    # Add title and axis names
    plt.title('Clique Distribution')
    plt.xlabel('Clique Size')
    plt.ylabel('Number of Cliques')
    plt.show()
    #
    return True
    
# *****************************************************************************
# *****************************************************************************
# *****************************************************************************
# *****************************************************************************
def maxClqSubG(OLS_CLQ,M, minCLQ):
    NCLQ=len(OLS_CLQ)
    subNodes=set()
    for m in range(0,NCLQ):
        tCLQ=set(OLS_CLQ[m])
        if len(tCLQ)>minCLQ:
            subNodes=subNodes|tCLQ # sub nodes
    #-- end for
    # construct graph from subnodes
    SG=nx.Graph()
    while(len(subNodes)>0):
        SG.add_node(subNodes.pop())
    #
    SG=updGStruct(SG)
    return SG
            
   
#----------------------------------------------------------------------------
# codelettes
#----------------------------------------------------------------------------
def updateMAXREADS(NSWEEPS, NV):
    #NTSWEEPS=NV*int(np.sqrt(NV))*NSWEEPS
    NTSWEEPS=NV*NSWEEPS
    return NTSWEEPS

'''
###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################
'''

'''
# ****************************************************************************
# MAIN PROGRAM: RANDOM GROWING OF H-GRAPH, OBTAIN LARGER CLIQUE
# ****************************************************************************
# Redundancy of JOBLIST and {NLS_CLQ, OLS_CLQ} => OMMITS !!!!!
'''    
if __name__ == '__main__':
    __spec__ = "ModuleSpec(name='builtins', \
               loader=<class '_frozen_importlib.BuiltinImporter'>)"
    '''
    # -----------------------------------------------------------
    # DEFINE PROBLEM
    # -----------------------------------------------------------
    '''
    #SUB_FACTOR=10 #int(M/4)    # sub-graph size = SUB_FACTOR*M
    # define H-order, initial parameters 
    kk=7 #9
    M = kk*4
    MAXITR=2*M
    # define number of sweeps/iteration
    NSWEEPS=1*1*1*M*10*1000
    # number of read or solutions in the NEAL
    NREADS=int(M/2)
    eps= 1e-9 #0.001/M
    # define number of cores
    NCPU=mp.cpu_count()
    NJOBS= NCPU-2 #int(3*NCPU/4)
    print('\nSolving H-Problem by', NJOBS, 'of total', NCPU, 'CPU cores')
    '''
    # -----------------------------------------------------------
    # PARAMETERS INITIALIZATION
    # -----------------------------------------------------------
    '''  
    fName_CLQ   = r'BKTRACK/CLQ'+str(M)+'.pkl'
    fName_G     = r'BKTRACK/G'+str(M)+'.pkl'
    fName_TRIAL = r'BKTRACK/TRIAL'+str(M)+'.pkl'
    fName_OLS_CLQ  = r'BKTRACK/OLS_CLQ'+str(M)+'.pkl'
    '''
    # -----------------------------------------------------------
    # INITIALIZATION OF DISCRETE STRUCTURES
    # -----------------------------------------------------------
    '''
    NDZ_svec=np.int_(np.ones(M)).tolist()
    NDZ_hex=svec2hex(NDZ_svec)
    ###############################################################
    F_SEED=0    # 0: first time running, 1: continu old job
    NV_LIM=0    # simulate failure in finding M-clique, continu next 
    ###############################################################
    THMinOLS=0.9
    
    if F_SEED>0:    # continue previous job
        #
        CLQ     = pk.load(open(fName_CLQ, 'rb'))
        G       = pk.load(open(fName_G, 'rb'))
        TRIAL   = pk.load(open(fName_TRIAL, 'rb'))
        OLS_CLQ = pk.load(open(fName_OLS_CLQ, 'rb'))
        
        #init subgraph
        minOCLQ=int(THMinOLS*len(CLQ)) # treshold of clique-size in subgraph
        SG=maxClqSubG(OLS_CLQ,M, minOCLQ)
    else:
        tvHex=svec2hex(genSHVector(M))
        # Init master Clique
        CLQ={''}
        CLQ.pop()
        #
        CLQ.add(tvHex)
        # Init master Graph
        G=nx.Graph()
        G.add_node(tvHex)
        # Init record of trial TRIAL<- {}
        TRIAL={()} #set()
        OLS_CLQ= [[tvHex]]
        # init subgraph
        SG=G.copy()

    # end-if-else --

    '''
    # ============================================================================
    # MAIN LOOP
    # ============================================================================
    '''
    itr=0
    #
    vgapE=[]            # record curve of energy
    vLenG=[]            # record Graph Size
    vLenSG=[]           # record Sub-Graph Size
    vMeanLenG=[]        # record mean degree of Graph
    vMeanLenSG=[]       # record mean degree of Sub-Graph
    vnjList=[]          #length of job list
    #
    vszG=[len(G.nodes())] # record graph size ---??? DUPLICATE ???  
    vszCLQ=[len(CLQ)]   # record size of max-clique
    currNV=len(CLQ)
    prevNV=currNV
    MAXBRANCH=2         # maximum branching allowed at per iteration per job
    # init random seed for reproducability
    NBRANCH=MAXBRANCH
    np.random.seed(17)
    #
    currNDG=len(G.nodes())
    currNDSG=len(SG.nodes())
    currMDG=mean_degree(G)
    currMDSG=mean_degree(SG)
    #
    vLenG.append(currNDG)
    vLenSG.append(currNDSG)
    vMeanLenG.append(currMDG)
    vMeanLenSG.append(currMDSG)

    #currNV=NV
    JOB_LIST=createJobList(SG,TRIAL)
    while ( (currNV<M-1-NV_LIM) and (itr<MAXITR) ) :
        print('===================================================')
        print('>>> FINDING -', currNV+1,'TH VECTOR OF -',M, 'ITER=', itr+1, 'of', MAXITR)
        print('===================================================')

        # calculate sweep numbers
        NTSWEEPS=updateMAXREADS(NSWEEPS, currNV)

        '''
        if ( (currNV>prevNV) or (len(JOB_LIST)<1)):
            JOB_LIST=createJobList(SG,TRIAL)
            print('Create new JOB_LIST. len=',len(JOB_LIST))
        else:
            print('Use existing JOB_LIST. len=',len(JOB_LIST))
        '''
        #
        JOB_LIST=createJobList(SG,TRIAL)
        #
        NJList=len(JOB_LIST)
        vnjList.append(NJList)  # for curve display
        #
        if NJList>0:
            #
            '''
            #********************************************************************
            # START OF PARALLEL PROCESSING
            #********************************************************************
            '''
            vParams, JOB_LIST=requestJobs(JOB_LIST, M, NJOBS, NTSWEEPS, NREADS)
            updateTRIAL(TRIAL, vParams)        
            # --------------------------------------------------------------------
            # spawn jobs and get results
            # --------------------------------------------------------------------
            #execute parallel jobs, then update G
            #
            print('Pool started ... !!!')
            pool = Pool(min(NJOBS, NJList)) #Pool(NJOBS)  ###=> Pool(min(NJOBS, NJList)) ???
            tV1=pool.map(findOrthoVec_PAR, vParams)
            pool.close()
            pool.join()  
            print('Pool closed ... !!!')
            # 
            G, CLQ = updateGRAPH(tV1, G, CLQ, NBRANCH)
            '''
            #********************************************************************
            # END OF PARALLEL PROCESSING
            #********************************************************************
            '''      
            #limit subgraph size
            THMinOLS=0.90
            NBRANCH=2   # 2/1: ?
            print('Limit sub graph and branching')
         
        else:
            '''
            # --
            # make number of branch flexible, to avoid zero job list to process
            # avoid collapsing job list in SUBGRAPH
            '''
            # enlarge subgraph size
            THMinOLS=0.95*THMinOLS   # 0.70       # = 0.8*ThMinOLS ???
            NBRANCH=int(M/4)        # = int(1.5*NBRANCH) ?
            print('Enlarge sub graph and branching, threshold=', THMinOLS)
        ## --- end doing job
        print('Size of tried orthoset:', len(TRIAL), \
              ', size of JOB_LIST:', len(JOB_LIST))

        # generate list of cliques from the graph G
        NLS_CLQ=list(nx.clique.find_cliques(G))
        OLS_CLQ=sortCLQS(NLS_CLQ)
        CLQ=maxCLQ(OLS_CLQ).copy()
        # update NV
        prevNV=currNV
        currNV=len(CLQ)
        print('Maximum CLIQUE:', prevNV,'->', currNV)
        
        minOCLQ=int(THMinOLS*len(CLQ)) # treshold of clique-size in subgraph

        '''
        # CONSTRUCT DENSE SUBGRAPH, after a particular iteration number
        '''
        TH_MAINGRAPH=M*2
        if ( len(G.nodes())> TH_MAINGRAPH or (currNV>M/4) ):     # or (currNV>M/4) :
            print('G reduced to SG')
            SG=maxClqSubG(OLS_CLQ,M, minOCLQ)
        else:
            print('G stay the same')
            SG=G.copy()
        # ---- v
        prevNDG=currNDG
        prevMDG=currMDG
        prevNDSG=currNDSG
        prevMDSG=currMDSG               
        # display updated graph        
        currNDG=len(G.nodes())
        currMDG=mean_degree(G)        
        currNDSG=len(SG.nodes())
        currMDSG=mean_degree(SG)
        print('#Nodes in G:', prevNDG, '->', currNDG, \
              '; Mean degree:',int(prevMDG), '->', int(currMDG))       
        print('#Nodes in SG:', currNDSG, '->', currNDSG, \
              '; Mean degree:',int(prevMDSG), '->', int(currMDSG))
        # -DEBUG--- memory size 
        #print('Mem size of G:', sys.getsizeof(G), ', SG:', sys.getsizeof(SG))
        
        '''
        # SAVE ALL PARAMS TO FILE: > CLQ, G, TRIAL, OLS_CLQ
        '''
        pk.dump(CLQ, open(fName_CLQ, 'wb+'))
        pk.dump(G, open(fName_G, 'wb+'))
        pk.dump(TRIAL, open(fName_TRIAL, 'wb+'))
        pk.dump(OLS_CLQ, open(fName_OLS_CLQ, 'wb+'))
        # --
        itr=itr+1
        vszG.append(len(G.nodes()))
        vszCLQ.append(currNV)
        
        
        #
        vLenG.append(currNDG)
        vLenSG.append(currNDSG)
        vMeanLenG.append(currMDG)
        vMeanLenSG.append(currMDSG)
        #
        
        #
        '''
        # plot curves NDG, NDSG,MDG, MDSG 
        '''
        fig, ax = plt.subplots()
        ax.plot(vLenG,'r-o', label='Size(G)') 
        ax.plot(vLenSG,'g-x', label='Size(SubG)')
        ax.plot(vMeanLenG,'b-*', label='Mean deg(G)')
        ax.plot(vMeanLenSG,'y-d', label='Mean deag(SubG)')
        ax.plot(vnjList,'k-.', label='Length of job list')
        #ax.axis('equal')
        leg = ax.legend();
        plt.title('Growth of graph')
        plt.xlabel('Iteration (t)')
        plt.ylabel('Size') 
        plt.show()
    ############
    if (currNV> M-2) and isOrthoSet(CLQ,M):
        print('Hadamard matrix of order',M, 'is found !!!')
    else:
        print('Hadamard matrix has not been found ...')
    
    '''
    # ============================================================================   
    # SAVE AND DISPLAY RESULTS
    # ============================================================================   
    '''   
    # write h-matrix if the search is successful
    fmatxname='HMTX/H'+str(M)+'.txt'
    if isOrthoSet(CLQ,M):
        HMAT=CLQ2svlist(CLQ,M)
        writeHMatrix(fmatxname,HMAT)
    # -- end-if
    print('Order of final CLQ:', len(CLQ))
    
    DD=matxIndicator(CLQ,M)
    imgplot = plt.imshow(abs(DD))
    plt.show()
    
    # draw graph ?
    #nx.draw(G,with_labels=True, font_weight='bold')
    
    # plot curves
    fig, ax = plt.subplots()
    ax.plot(vszG,'r-o', label='Graph size') 
    ax.plot(vszCLQ,'b-*', label='Max-Clique size')
    #ax.axis('equal')
    leg = ax.legend();
    plt.title('Growth of graph')
    plt.xlabel('Iteration (t)')
    plt.ylabel('Size') 
    plt.show()

"""    
    # display histogram of cliques
    #
    NBIN=10
    tbins=[]
    dbin=1
    for m in range (0,M,dbin):
        tbins.append(m)
    #
    num_bins=M
    sdat=listSCLQ(OLS_CLQ)
    n, bins, patches = plt.hist(sdat, num_bins, facecolor='blue', alpha=0.5)
    # Add title and axis names
    plt.title('Clique Distribution')
    plt.xlabel('Clique Size')
    plt.ylabel('Number of Cliques')
    plt.show()
    #
    sdat1=listSCLQ(list(TRIAL))
    num_bins=M
    n, bins, patches = plt.hist(sdat1, num_bins, facecolor='blue', alpha=0.5)
    # Add title and axis names
    plt.title('Distribution of Attempts')
    plt.xlabel('Clique Size (before attempt)')
    plt.ylabel('Number of Attemps')
    plt.show()
"""
