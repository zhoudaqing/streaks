import numpy as np
import pandas as pd
import math, itertools, glob
import pickle
import random
from collections import OrderedDict

def nchoosek(n,k):
    return math.factorial(n)/(math.factorial(n-k)*math.factorial(k))

# function that interchanges 1 and 0
def f(x):
    return (x-1)*(x-1)

histProbS = {}      # global history of ProbS() computations
histE = {}          # global history of E() computations
histPart = {}
histProbSp = {}
histProbSAtleastp = {}

# computes the probability that a sequence of length n with bernuilli variable
# with probablity p contains EXACTLY m streaks of length k or more
# hitting recursive limit-- can make more cases more explicit?
# can break it in half?
# build up the dictionary?
def ProbSp(k,n,p,m):
    if n == 0 and m == 0:
        return 1
    if n < k*m + m - 1 or n < 0 or m < 0:
        return 0
    if n == k*m + m - 1:
        return (p**(k*m))*((1-p)**(m-1))
    if (k,n,p,m) in histProbSp:
        return histProbSp[(k,n,p,m)]

    # handle the case where there is no first zero
    if (m == 1 and k <= n) or (m == 0 and k > n):
        ret = p**n
    else:
        ret = 0

    for j in range(1,k+1):
        ret += (p**(j-1))*(1-p)**(n >= j)*ProbSp(k,n-j,p,m)  # need to worry about boundary case of k=n
    for j in range(k+1,n+1):
        ret += (p**(j-1))*(1-p)*ProbSp(k,n-j,p,m-1)
    histProbSp[(k,n,p,m)] = ret
    return ret

# given an array n consisting of the number of shots of the various games, an array
# n1 of the number of makes in each game, this computes the probability that
# we get a number of k-streaks greater than or equal to m (a fixed non-negative integer)
def Pvalp(k,n,p,m):
    # first we get all the tuples whose sum is bigger than or equal to m
    M = partitions(k,n,m)
    ret = 0
    for ms in M:
        ret += np.prod([ProbSp(k,n[i],p,ms[i]) for i in range(len(ms))])
    return ret

# computes the probability that a sequence of length n with bernuilli variable
# with probablity p contains AT LEAST m streaks of length k or more
def ProbSAtleastp(k,n,p,m):
    if m <= 0:
        return 1
    if n < k*m + m - 1 or n < k:
        return 0
    if n == k*m + m - 1:
        return (p**(k*m))*((1-p)**(m-1))
    if (k,n,p,m) in histProbSAtleastp:
        return histProbSAtleastp[(k,n,p,m)]

    # handle the case where there is no first zero
    if k <= n and m <=1:
        ret = p**n
    else:
        ret = 0

    for j in range(1,k+1):
        ret += (p**(j-1))*(1-p)**(n >= j)*ProbSAtleastp(k,n-j,p,m)  # need to worry about boundary case of k=n
    for j in range(k+1,n+1):
        ret += (p**(j-1))*(1-p)*ProbSAtleastp(k,n-j,p,m-1)
    histProbSAtleastp[(k,n,p,m)] = ret
    print(n,m)
    return ret

# computes the probability that a sequence of length n with n1-many 1s
# contains exactly m streaks of length k or more
# also try to build up this dictionary
def ProbS(k,n,n1,m):
    if n1 > n or k > n or k*m > n1 or k > n1:
        if m == 0:
            return 1
        else:
            return 0 # should really be an error/exception?
    if n==n1:
        if m==1:
            return 1
        else:
            return 0
    if (k,n,n1,m) in histProbS:
        return histProbS[(k,n,n1,m)]
    ret = 0
    for j in range(1,k+1):
        ret += ProbS(k,n-j,n1-j+1,m)*nchoosek(n-j,n1-j+1)/nchoosek(n,n1)
    for j in range(k+1,n1+2):
        ret += ProbS(k,n-j,n1-j+1,m-1)*nchoosek(n-j,n1-j+1)/nchoosek(n,n1)
    # need to look at the case n = n1 + 1? or case m = 0
    histProbS[(k,n,n1,m)] = ret
    return ret


# given an array n consisting of the sizes of the various games, an array
# n1 of the number of makes in each game, this computes the probability that
# we get a number of k-streaks greater than or equal to m (a fixed non-negative integer)
def Pval(k,n,n1,m):
    # first we get all the tuples whose sum is bigger than or equal to m
    M = partitions(k,n1,m)
    ret = 0
    for ms in M:
        ret += np.prod([ProbS(k,n[i],n1[i],ms[i]) for i in range(len(ms))])
    return ret

# returns all possible arrays a of length len(n1) such that sum(a) >= m and k*a[i] <= n1[i]
def partitions(k,n1,m):
    # should start off by replacing each element of n1 with k times the floor of it divided by k
    #m = math.ceil(m)
    n1 = [k*math.floor(x/k) for x in n1]
    if (k,tuple(n1),m) in histPart:
        return histPart[(k,tuple(n1),m)]
    if len(n1) == 1:
        ret = [[x] for x in range(m, math.floor(n1[0]/k) + 1)]
        histPart[(k,tuple(n1),m)] = ret
        return ret

    p = []
    # i = number of streaks for first game
    for i in range(math.floor(n1[0]/k) + 1):
        subparts = partitions(k,n1[1:],max(0,m-i))
        for sp in subparts:
            p.append([i] + sp)
    histPart[(k,tuple(n1),m)] = p
    return p

def PvalMC(k,n,n1,m,N):
    count = 0
    for j in range(N):
        streaks = 0
        for i in range(len(n)):
            # streaks += CountStreaksAtleast(k,genranddata1(n[i],n1[i],1)[0].tolist())
            streaks += CountStreaksAtleast(k,genranddata(n[i],n1[i]))
        if streaks >= m:
            count += 1
    return count/N

def PvalMC2(k,n,n1,m,N):
    count = 0
    for j in range(N):
        streaks = 0
        randata = genranddata(sum(n),n1)
        i = 0
        leng = 0

        for i in range(len(n)):
            gshots = randata[sum(n[:i]):sum(n[:i])+n[i]] # shots for ith game
            streaks += CountStreaksAtleast(k,gshots)
            #print(gshots)
            i += n[i]
            leng += len(gshots)

        if streaks >= m:
            count += 1
        # print(sum(n),leng)
    return count/N

# generate a random list of length a with b many 1's and n-n1 many 0's
def genranddata(a,b):
    ret = [0]*a
    # randomly select n1 integers from 0 to n-1 to serve as the indices
    indices = random.sample(range(a),b)
    for i in indices:
        ret[i] = 1
    return ret

def genranddata1(n,n1,N):
    ret = []
    while len(ret) < N:
        a = np.random.binomial(1,.5,n)
        if sum(a) == n1:
            ret.append(a)
    return ret

# function that counts how many r streaks are in an array a
# CHECK THIS NOW SINCE CHANGED IT
def CountStreaks(k,a):
    t = 0
    a = [0] + a + [0]
    for i in range(1,len(a)-1):
        if (a[i-1:i+k+1] == ([0] + [1]*k + [0])):
            t += 1
            i += k+1
    return t

# this is very inefficient, replaced below
# def CountStreaksAtleast(k,a):
#     ret = 0
#     for i in range(k,len(a)+1):
#         ret += CountStreaks(i,a)
#     return ret

# assume k >= 1
def CountStreaksAtleast(k,a):
    na = np.array(a)
    idx = np.where(na == 0)[0].tolist()   # get indices of zeros
    # take care of boundary cases
    idx.append(len(a))
    idx.insert(0,-1)
    ret = 0
    for i in range(len(idx)-1):
        if idx[i+1] - idx[i] >= k+1:
            ret += 1
    return ret


def EAtleast(k,n,p):
    ret = 0
    for i in range(k,n+1):
        ret += E(i,n,p)
    return ret

# check this
# this should be the expected number of streaks of length exactly k
def E(r,n,p):
    if r > n:
        return 0
    elif r==n:
        return p**n
    if (r,n,p) in histE:
        return histE[(r,n,p)]
    else:
        ret = p**r*(1-p) + (r==n)*(p**n)
        for j in range(1,n+1):
            ret += E(r,n-j,p)*(p**(j-1))*(1-p)
    histE[(r,n,p)] = ret
    return ret

# takes in the data of the shots throughout a collection of games (shotdata) and returns
# the number of streaks of length at least r
def PlayerStreaks(shotdata, r):
    ns = 0  # number of streaks
    for g in list(OrderedDict.fromkeys(shotdata['game_date'])): #iterate through all the games
        shots = shotdata.loc[shotdata['game_date'] == g]['shot_made_flag'].tolist() # get the shots from game g
        for i in range(r,len(shots)+1):
            ns += CountStreaks(i,shots)
    return ns

# takes in the data of the shots throughout a colleciton of games (shotdata) and returns
# the expected number of streaks of length at least r
def ExpectedPlayerStreaks(shotdata, r):
    # compute players field goal % from the given data
    fgp = sum(shotdata['shot_made_flag'].values.tolist()) / len(shotdata['shot_made_flag'].values.tolist())
    ens = 0
    for g in list(OrderedDict.fromkeys(shotdata['game_date'])): #iterate through all the games
        s = len(shotdata.loc[shotdata['game_date'] == g]['shot_made_flag'].tolist()) # get the number of shots taken in game g
        ens += EAtleast(r,s,fgp)
    return ens

# takes in the data of the shots throughout a colleciton of games (shotdata) and returns
# the expected number of streaks of length at least r
# now calcuate the expected players streaks by using fgp per season instead of all-time
def ExpectedPlayerStreaks2(shotdata, r):
    # compute players field goal % from the given data
    ens = 0
    for s in list(OrderedDict.fromkeys(shotdata['season'])): #iterate through all the seasons
        seasondata = shotdata.loc[shotdata['season'] == s]
        ens += ExpectedPlayerStreaks(seasondata,r)
    return ens

# create csv file of results for hot or cold streak of length k. horc should have value 'hot' or 'cold'
def createtable(k, horc, directory, resultsdir, N=10000):
    playernames, streaks, estreaks, pvals1, pvals2 = [], [], [], [], []  # arrays of player names, streaks, expected streaks, and p-value

    for filename in glob.glob(directory + '*.csv'):
        data = pd.read_csv(filename, usecols=['name', 'game_date', 'period', 'minutes_remaining', 'seconds_remaining', 'shot_made_flag', 'season'])
        data = data.sort_values(['game_date', 'period', 'minutes_remaining', 'seconds_remaining'], ascending=[True, True, False, False])
        if horc == 'cold':
            data['shot_made_flag'] = f(data['shot_made_flag']) # interchange 1's and 0's
        ps = PlayerStreaks(data,k)
        eps = ExpectedPlayerStreaks2(data,k)
        streaks.append(ps)
        estreaks.append(eps)
        pvals1.append(1-calcPvalMC(data,ps,k,N))
        pvals2.append(1-calcPvalMC2(data,ps,k,N))
        playernames.append(data['name'][0])
        print(data['name'][0] + ' ' + str(ps) + ' ' + ' ' + str(eps) + ' ' + str(pvals1[-1]) + ' ' + str(pvals2[-1]))

    table = {'Name': playernames, 'Streaks': streaks, 'Expected Streaks': estreaks, 'p-val1': pvals1, 'p-val2': pvals2}
    tabledf = pd.DataFrame(data=table)
    #tabledf = tabledf.sort_values('p-value', ascending=True)

    tabledf.to_csv(resultsdir + horc + str(k) + str(N) + '.csv')

#N number of simulations to use in the Monte Carlo approximation of the p-value (10000 before?)
def calcPvalMC(shotdata,m,k, N=10000):
    n,n1 = [],[]
    for g in list(OrderedDict.fromkeys(shotdata['game_date'])):
        n.append(len(shotdata.loc[shotdata['game_date'] == g]['shot_made_flag'].tolist()))
        n1.append(sum(shotdata.loc[shotdata['game_date'] == g]['shot_made_flag'].tolist()))
    return PvalMC(k,n,n1,m,N)

# same as above except now shuffle the total number of makes in the total season
# instead of per game
def calcPvalMC2(shotdata,m,k,N=10000):
    n = []
    for g in list(OrderedDict.fromkeys(shotdata['game_date'])):
        n.append(len(shotdata.loc[shotdata['game_date'] == g]['shot_made_flag'].tolist()))
    n1 = sum(shotdata['shot_made_flag'])
    return PvalMC2(k,n,n1,m,N)


# sim flag is whether or not to simulate the computation
def calcPval(shotdata,k,d,sim=False,N=10):
    n,n1,shots = [],[],[]
    for g in list(OrderedDict.fromkeys(shotdata['game_date'])):
        gshots = shotdata.loc[shotdata['game_date'] == g]['shot_made_flag'].tolist()
        shots.append(gshots)
        n.append(len(gshots))
        n1.append(sum(gshots))
    ret = 0
    retMC = 0
    i = 0
    while (i + d < len(n)):
        subn1, subn = n1[i:i+d], n[i:i+d]
        streaks = sum([CountStreaksAtleast(k,s) for s in shots[i:i+d]])      # number of streaks
        print(subn,subn1,streaks,Pval(k,subn,subn1,streaks))
        ret += Pval(k,subn,subn1,streaks)
        retMC += PvalMC(k,subn,subn1,streaks,N)
        i+=d
    # print(ret, retMC)
    #print (retMC/math.floor(len(n)/d), ret/math.floor(len(n)/d))
    return ret/math.floor(len(n)/d)

def loadstuff():
    for n in range(1000):
        for n1 in range(n+1):
            for m in range(math.ceil(n1/3)+1):
                ProbS(3,n,n1,m)

    np.save('histProbS.npy', histProbS)

    pickle.dump(histProbS, open("histProbS.p", "wb"))

# data = pd.read_csv('data/2500plus/nba_savant202083.csv', usecols=['name', 'game_date', 'period', 'minutes_remaining', 'seconds_remaining', 'shot_made_flag', 'season'])
# data = data.sort_values(['game_date', 'period', 'minutes_remaining', 'seconds_remaining'], ascending=[True, True, False, False])
# n,n1,shots = [],[],[]
# for g in list(OrderedDict.fromkeys(data['game_date'])):
#     gshots = data.loc[data['game_date'] == g]['shot_made_flag'].tolist()
#     shots.append(gshots)
#     n.append(len(gshots))
#     n1.append(sum(gshots))
#
# shotsflat = [j for i in shots for j in i]
#
# i = 0
# d = 1500
# fgp = []
# k = 3
#
# while(i+d < len(shotsflat)):
#     subshots = shotsflat[i:i+d]
#     fgp = sum(subshots)/len(subshots)
#     s = CountStreaksAtleast(k,subshots)
#     print(s,fgp,Pvalp(k,[len(subshots)],fgp,s))
#     i+=d

# TRY USING BAYES TO COMPUTE EXPLICITLY  then look at prob of a streak not occuring through a given index

# while(i+d < len(n)):
#     fgp = sum(n1[i:i+d])/sum(n[i:i+d])
#     s = sum([CountStreaksAtleast(k,gs) for gs in shots[i:i+d]])
#     print(s, fgp, Pvalp(k,n[i:i+d],fgp,s))
#     i+=d

# print(calcPval(data,3,30))
#
# problem is that the number of streaks over a small amount of games is small. shuffle over games?
# change this so that not doing a permutation test.  given e.g. n = [5,5,5,5] and n1 = [5,5,0,0]
# of course 2 streaks of 5 is definite even though they shot 50%
#
# for s in ['2500plus', 'threes']:
#     DIR = 'data/' + s + '/'
#     RESULTSDIR = 'data/results/' + s + '/'
#     for hc in ['hot', 'cold']:
#         for r in [5,4,3]:
#             print(r)
#             createtable(r, hc, DIR, RESULTSDIR)
