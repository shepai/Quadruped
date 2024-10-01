#skeleton class for all the models
class agent:
    def __init__(self):
        self.positions=[0 for i in range(12)]
        self.genotype=None
    def get_positions(self):
        return self.positions
    def set_genotype(self,values):
        self.geno=values
    def get_genotype(self):
        return self.geno
    