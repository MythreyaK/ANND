class Optimizer:
    class SGDMomentum:
        def __init__(self, momentum=0.9, regularizer=0.1):
            self.m = momentum
            self.prevDws = None
            self.prevDbs = None
            self.type="SGD (M) M={0} ".format(self.m)

        def __call__(self, netInst, dws, dbs):
            # The instance of the network is also
            # passed so that we have access to the
            # weights and can consequently modify them
            if self.prevDbs is None:
                self.prevDws = dws
                self.prevDbs = dbs
            else:
                for i in range(len(netInst.weights) - 1, 0, -1):
                    netInst.weights[i] -= (self.m * self.prevDws[i] +
                                           (1 - self.m) * netInst.layers[i].lr * dws[i])
                    netInst.layers[i].bias -= (self.m * self.prevDbs[i] + (
                        1 - self.m) * netInst.layers[i].lr * dbs[i])
                self.prevDws = dws
                self.prevDbs = dbs

    class SGD:
        def __init__(self):
            pass
            self.type="SGD "

        def __call__(self, netInst, dws, dbs):
            for i in range(len(netInst.weights) - 1, 0, -1):
                netInst.weights[i] -= dws[i]*netInst.layers[i].lr
                netInst.layers[i].bias -= dbs[i]*netInst.layers[i].lr

    class Adam:
        def __init__(self, beta1=0.9, beta2=0.99):
            self.b1 = beta1
            self.b2 = beta2
            self.eps = 10**-5
            self.b_mt = None
            self.b_vt = None
            self.w_mt = None
            self.w_vt = None
            self.type = "Adam B1={0}, B2={1} ".format(
                self.b1, self.b2
            )

        def __call__(self, netInst, dws, dbs):
            if self.b_vt is None:
                self.b_mt   = [0]*len(netInst.weights)
                self.b_vt   = [0]*len(netInst.weights)
                self.b_mhat = [0]*len(netInst.weights)
                self.b_vhat = [0]*len(netInst.weights)

                self.w_mt   = [0]*len(netInst.weights)
                self.w_vt   = [0]*len(netInst.weights)
                self.w_mhat = [0]*len(netInst.weights)
                self.w_vhat = [0]*len(netInst.weights)

            else:
                for i in range(len(netInst.weights) - 1, 0, -1):
                    self.w_mt[i]  =  self.b1 * self.w_mt[i]  +  (1 - self.b1)*dws[i]
                    self.w_vt[i]  = (self.b2 * self.w_vt[i]) + ((1 - self.b2)*(dws[i]**2))
                    # self.w_mhat[i]= self.w_mt[i]/(1 - self.b1)
                    # self.w_vhat[i]= self.w_mt[i]/(1 - self.b2)

                    netInst.weights[i] -= (netInst.layers[i].lr * self.w_mt[i])/(self.eps + self.w_vt[i]**0.5)
                    # netInst.weights[i] -= (netInst.layers[i].lr * self.w_mt[i])/(self.eps + self.w_vt[i]**0.5)


                    self.b_mt[i]  =  self.b1 * self.b_mt[i]  +  (1 - self.b1)*dbs[i]
                    self.b_vt[i]  = (self.b2 * self.b_vt[i]) + ((1 - self.b2)*(dbs[i]**2))
                    # self.b_mhat[i]= self.b_mt[i]/(1 - self.b1)
                    # self.b_vhat[i]= self.b_mt[i]/(1 - self.b2)

                    netInst.layers[i].bias -= (netInst.layers[i].lr * self.b_mt[i])/(self.eps + self.b_vt[i]**0.5)
                    # netInst.layers[i].bias -= (netInst.layers[i].lr * self.b_mt[i])/(self.eps + self.b_vt[i]**0.5)