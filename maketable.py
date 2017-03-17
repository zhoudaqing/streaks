import main

for horc in ['cold','hot']:
    for k in [5,4,3]:
        for DIR in ['2500plus','threes']:
            main.createtable(k,horc,'data/' + DIR + '/','data/results/' + DIR + '/',10000) 
