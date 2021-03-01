import numpy as np

'''
Diagonal model of evidence for mathematically simplicity
input: key of evidence t
returns: sensor model in matrix form considering evidence is True/False
'''
def O_t(ev):
    if ev:
        return np.array([[0.75, 0],[0, 0.2]])
    else:
        return np.array([[0.25, 0],[0, 0.8]])
'''
Vector model of evidence
input: key of evidence t
returns: sensor model in vector form considereing evidence is True/False
'''
def O(ev):
    if ev:
        return np.array([0.75, 0.2])
    else:
        return np.array([0.25, 0.8])    

def forward(prior, evidence, transition, t0, t1, prnt):
    f = prior 
    last_ev = list(evidence)[-1]
    first_ev = list(evidence)[0]
    if t1 == 0:
        return f                                                # No transition 
    for i in range(t0, t1 + 1):
        try:
            #Simplified matrix alg. from AIMA eq(15.12)
            prob = np.dot(np.dot(O_t(evidence[i]), np.transpose(transition)), f)  
            if prnt : print('P(X_{}|e_{}:{}): '.format(i, first_ev, i))
        except:                                                  #When no more evidence --> prediction
            prob = np.dot(np.transpose(transition), f)
            if prnt : print('P(X_{}|e_{}:{}): '.format(i, first_ev, last_ev))
        prob = normalize(prob)
        f = prob
        if prnt: print(f, "\n")
    return prob

def backward(evidence, transition, k2, k1, prnt):
    b = np.array([[1.0],[1.0]])
    last_ev = list(evidence)[-1]
    first_ev = list(evidence)[0]
    for i in range(k2-1, k1-1, -1):
        try:
            #Simplified matrix alg. from AIMA eq(15.13)
            prob = np.dot(np.dot(transition,O_t(evidence[i+1])), b)
            if prnt : print('Backward: P(X_{}|e_{}:{}): '.format(i, first_ev, k2))
        except:                                                 #Empty evidence
            prob = np.dot(transition, b)
            if prnt : print('Backward: P(X_{}|e_{}:{}): '.format(i-1, first_ev, k2))
        b = prob
        if prnt : print(prob, "\n")
    return prob


def normalize(d):
    """
    Returns: normalized distribution. From Assignment 1
    """
    return np.array(list(i/sum(d) for i in d)) 

def forward_backward(evidence, transition, prior, t0): 
    sv = []
    for i in range(len(evidence)):
        f = forward(prior, evidence, transition, t0, i, False)
        b = backward(evidence, transition, len(evidence), i, False)
        sv.append(normalize(f*b))
    return sv

'''
Implementation from pseudocode from https://en.wikipedia.org/wiki/Viterbi_algorithm, but with simplified
matrix model
'''
def viterbi(evidence, transition, prior, dim):
    X = []                                                   #Most likely hidden stat sequence/argmax
    trellis = np.zeros((dim,len(evidence)))                  #To hold p of each state given each observation
    m = np.dot(O_t(evidence[1]),np.dot(np.transpose(transition), prior))
    m.reshape(dim,1)
    trellis[:, 0] = m[:, 0]

    for o in range(1, len(evidence)):                       #Adding pmax for all o in trellis
        max_prob = np.maximum((trellis[:, o-1] * transition)[0], (trellis[:, o-1] * transition)[1])
        ev = O(evidence[o+1])
        m = max_prob * ev
        trellis[:, o] = m
    for i in range(len(trellis[0])):
        X.append(trellis[:, i][0] > trellis[:, i][1])        #adding argmax to X
    return X


states = (True, False)
prior = np.array([[0.5],[0.5]]) #Prior state
evidence = {                    #Evidence True if birds, False if no birds
    1 : True, 
    2 : True, 
    3 : False, 
    4 : True,
    5 : False, 
    6 : True
}
'''
    Transition model in matrix form
    P(X_t|X_t-1)
    [[0.8 0.2]
     [0.3 0.7]]
'''
T = np.array([[0.8,0.2],[0.3,0.7]])

def problem1b():
    print("Problem 1b: \n")  
    forward(prior, evidence, T, 1, 6, True)

def problem1c():
    print("Problem 1c: \n")  
    p = forward(prior, evidence, T, 1, 6, False)
    forward(p, evidence, T, 7, 30, True)

def problem1d():
    print("Problem 1d: \n")
    sv = forward_backward(evidence, T, prior, 0)
    for i in range(6):
        print("P(X_{}|e1:{})".format(i, len(evidence)),)
        print(sv[i], "\n")

def problem1e():
    seq = viterbi(evidence, T, prior, 2)
    print("Problem 1e: \n")
    print("Most likely hidden state sequence: \n", seq)


if __name__ == "__main__":
    problem1b()
    problem1c()
    problem1d()
    problem1e()



