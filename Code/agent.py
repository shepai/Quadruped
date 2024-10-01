#skeleton class for all the models
import numpy as np
class agent:
    def __init__(self):
        self.positions=[0 for i in range(12)]
        self.genotype=None
    def get_positions(self,values):
        return self.positions
    def set_genotype(self,values):
        self.geno=values
    def get_genotype(self):
        return self.geno
    def mutate(self,p=0.1):
        self.geno+=np.random.normal(0,5*p,self.geno.shape)
    