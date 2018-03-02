net = LayeredNet([28*28,15,8,10])

save = net.synapse
prev = 0
count = 0
succ = []
shuffled = np.random.permutation(np.arange(0,samples,10))
frac = 1/60 # how much of the set are we going to train on
for i in range(5):
    net.grow(50)
    frac = frac * 2
    for i in shuffled[0:round(samples*frac)]:
        save = net.synapse
        lab = np.array(np.matrix(label[i:i+10,:]))
        for j in range(1): # Do multiple steps of learning
            net.learn(numbers[i:i+10],lab)
        net.success_rate(numbers[0:round(samples*frac)],label[0:round(samples*frac)])
        choice = "Improvement found"
        heat = (samples*frac-i)/(samples*frac)
        curr = net.success_probability
        if(prev > curr):
            if random()/(prev-curr) > heat: # todo: needs proper scheme for weighting really bad learning cases
                prev = curr
                choice = "Choosing worse"
            else:
                net.synapse = save
                choice = "Not choosing worse"
        else:
            prev = net.success_probability
        succ.append(prev)
        print("Iter:", i, choice, prev, end='\r')
