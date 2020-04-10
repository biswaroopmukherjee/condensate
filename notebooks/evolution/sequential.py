from condensate.chamber import Environment
from condensate.evolution import Evolve

class Sequential:
    '''An experimental sequence of evolutions of the condensate. 

    '''


    def __init__(self, environment):
        self.environment = environment
        self.blocks=[]

    @property
    def blocks(self):
        return self.blocks

    def add(self, evolve):
        '''Add an evolution block'''
        if not isinstance(evolve, Evolve):
            return TypeError('All evolution steps need to be of the condensate.evolution.Evolve type')
        self.blocks.append(evolve)

    def run(self, live=True, saveevery=200, saveframes=False, savemovie=True)):
        movies = []
        for bix in range(len(self.blocks)):
            moviename = self.blocks[bix].run(environment=self.environment, live=live, saveevery=saveevery, saveframes=saveframes, savemovie=savemovie)
            movies.append(moviename)
        
        # if savemovie:
        #     data.join(movies)


        

        