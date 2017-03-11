import numpy as np
import pandas as pd
import math, itertools, glob
from collections import OrderedDict

# function that interchanges 1 and 0
def f(x):
    return (x-1)*(x-1)

histProbS = {}      # global history of ProbS() computations
histE = {}          # global history of E() computations

# computes the probability that a sequence of length n with n1-many 1s
# contains exactly m streaks of length k or more
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

# give an array n consisting of the sizes of the various games, an array
# n1 of the number of makes in each game, this computes the probability that
# we get a number of k-streaks greater than or equal to m (a fixed non-negative integer)
def Pval(k,n,n1,m):
    g = len(n)      # number of games

    # first we get all the tuples whose sum is bigger than or equal to m
    L = [range(0,math.ceil(x/k)+1) for x in n1]
    M = list(itertools.product(*L))
    i = 0
    while i < len(M):
        if sum(M[i]) < m:
            M.remove(M[i])
        else:
            i+=1
    ret = 0
    for ms in M:
        ret += np.prod([ProbS(k,n[i],n1[i],ms[i]) for i in range(len(ms))])
    return ret

def PvalMC(k,n,n1,m,N):
    count = 0
    for j in range(N):
        streaks = 0
        for i in range(len(n)):
            streaks += CountStreaksAtleast(k,genranddata1(n[i],n1[i],1)[0].tolist())
        if streaks >= m:
            count += 1
    return count/N

def genranddata1(n,n1,N):
    ret = []
    while len(ret) < N:
        a = np.random.binomial(1,.5,n)
        if sum(a) == n1:
            ret.append(a)
    return ret

# function that counts how many r streaks are in an array a
def CountStreaks(r,a):
    t = 0
    a = [0] + a + [0]
    for i in range(1,len(a)-1):
        if (a[i-1:i+r+1] == ([0] + [1]*r + [0])):
            t += 1
    return t

def CountStreaksAtleast(r,a):
    ret = 0
    for i in range(r,len(a)+1):
        ret += CountStreaks(i,a)
    return ret

def EAtleast(r,n,p):
    ret = 0
    for i in range(r,n+1):
        ret += E(i,n,p)
    return ret

# check this
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

# create csv file of results for hot or cold streak of length r0. horc should have value 'hot' or 'cold'
def createtable(r0, horc, directory, resultsdir):
    playernames, streaks, estreaks, pvals = [], [], [], [] # arrays of player names, streaks, expected streaks, and p-value

    for d in glob.glob(directory + '*.csv'):
        data = pd.read_csv(d, usecols=['name', 'game_date', 'period', 'minutes_remaining', 'seconds_remaining', 'shot_made_flag', 'season'])
        data = data.sort_values(['game_date', 'period', 'minutes_remaining', 'seconds_remaining'], ascending=[True, True, False, False])
        if horc == 'cold':
            data['shot_made_flag'] = f(data['shot_made_flag']) # interchange 1's and 0's
        ps = PlayerStreaks(data,r0)
        eps = ExpectedPlayerStreaks2(data,r0)
        streaks.append(ps)
        estreaks.append(eps)
        pvals.append(1-calcPvalMC(data,ps,r0))
        playernames.append(data['name'][0])
        print(data['name'][0] + ' ' + str(ps) + ' ' + ' ' + str(eps) + ' ' + str(pvals[-1]))

    table = {'Name': playernames, 'Streaks': streaks, 'Expected Streaks': estreaks, 'p-value': pvals}
    tabledf = pd.DataFrame(data=table)
    tabledf = tabledf.sort_values('p-value', ascending=True)

    tabledf.to_csv(resultsdir + horc + str(r0) + '.csv')

def calcPvalMC(shotdata,m,k):
    n,n1 = [],[]
    N = 100   # number of simulations to use in the Monte Carlo approximation of the p-value (10000 before?)
    for g in list(OrderedDict.fromkeys(shotdata['game_date'])):
        n.append(len(shotdata.loc[shotdata['game_date'] == g]['shot_made_flag'].tolist()))
        n1.append(sum(shotdata.loc[shotdata['game_date'] == g]['shot_made_flag'].tolist()))
    return PvalMC(k,n,n1,m,N)

def calcPval(shotdata,m,k):
    n,n1 = [],[]
    for g in list(OrderedDict.fromkeys(shotdata['game_date'])):
        n.append(len(shotdata.loc[shotdata['game_date'] == g]['shot_made_flag'].tolist()))
        n1.append(sum(shotdata.loc[shotdata['game_date'] == g]['shot_made_flag'].tolist()))
    return Pval(k,n,n1,m)

for s in ['2500plus', 'threes']:
    DIR = 'data/' + s + '/'
    RESULTSDIR = 'data/results/' + s + '/'
    for hc in ['hot', 'cold']:
        for r in [5, 4, 3]:
            print(r)
            createtable(r, hc, DIR, RESULTSDIR)
