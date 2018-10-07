import numpy as np

#data is 2d (2nd dimension is for bias unit, all are 1)
#Each sample is in columns now
data = [[-4,-3,-2,-1,0,1,2,3,4,5,6],[1,1,1,1,1,1,1,1,1,1,1]]; #2 x 11
label = [0,0,0,0,0,1,1,1,1,1,1];
X=np.array(data);
Y=np.array(label);

#initialize weights (instead of random, we use predefined weight)
totEpoch=5; 
eta=0.5;                                                #learning rate
weights=[2,-8];                                         #line is y=2x-8    (1x2)
W=np.array(weights);

#plot current classifier status
plotLearning(X,Y,W)

#Get classifier decision on sample data
D=stepActivation(X,W);

for epoch in range(totEpoch):
    for sample in range(Y.shape[0]):
        #update w using delta rule
        W=W + eta * (Y[sample]-D[sample]) * X[0,sample]; #update both wi and bias
    D=stepActivation(X,W);
        #print(W)
    
    plotLearning(X,Y,W)
        