
import cobra
import random
import numpy as np
from time import sleep
from tqdm import tqdm
from datetime import datetime
from collections import OrderedDict
from matplotlib import pyplot as plt
from multiprocessing import Pool
from pathlib import Path


class QuantitativeMutation:
    def __init__(self, model, 
                 biomass_id    = 'biomass', 
                 system_rxn_id = 'ATPM',
                 extra_genes   = [],
                 verbose       = False):
        '''
        model - It can either be a path to a .json file containing the model, or
                an instance of a cobra.core.model.Model class, or (not working)
                a standard name to load the model from the internal ones.
                
        biomass_id    - In some models the biomass reaction is named differently. For example,
                        in yeast8 the biomass reaction is r_2111 and it is called "growth". In 
                        that case specify biomass_id = 'r_2111'.
                        
        system_rxn_id - List of reaction ids that will be ignored in all bound-related computations.
                        For example, the ATP maintenance rate reaction.
   
        extra_genes   - Some models don't have in model.genes all the gene names that 
                        appear in the gene reaction rules. e.g. s0001 in ecoli_core. If this is
                        the case, an error will pop up when calling apply_dosage().
                      
        verbose - Prints out whatever is hampening. Useful for quick debugging.    
        '''
        
        self.VERBOSE     = verbose
        
        
        ####### Load model #######
        if type( model) == cobra.core.model.Model:
            self.model      = model
            self.MODEL_NAME = model.id
            if self.VERBOSE: print('Loaded model %s from user input.' % self.MODEL_NAME)
                
        elif type(model) == str:
            self.model = cobra.io.load_json_model( model )
            self.MODEL_NAME = self.model.id
            if self.VERBOSE: print('Loaded model %s from user file.' % self.MODEL_NAME)
        
        
        ####### Reactions #######
        # Store separately the biomass reaction
        self.BIOMASS_ID = [rxn.id for rxn in self.model.reactions if biomass_id in rxn.id.lower() ][0]
        if self.VERBOSE: print('Biomass reaction id is %s' % self.BIOMASS_ID)
        
        
        # Store the exchange reactions as either having a tag, or as a user-defined list of reactions
        self.EX_RXNS = np.array( [ rxn.id for rxn in self.model.exchanges] )

        
        # Store the system reactions as either having a tag, or as a user-defined list of reactions
        if type(system_rxn_id) == str:
            self.SYS_RXNS  =  np.array( [ rxn.id for rxn in self.model.reactions if system_rxn_id in rxn.id ] )
           
        elif type(system_rxn_id)== list:
            self.SYS_RXNS  =  np.array( system_rxn_id )
      
        
        # Store non exchange reactions as all that are neither biomass, exchange nor system-rxns
        ignore_rxns = np.hstack( (self.BIOMASS_ID, self.EX_RXNS, self.SYS_RXNS))
        self.NEX_RXNS = np.array( [ rxn.id for rxn in self.model.reactions if rxn.id not in ignore_rxns] )
        
        
        # Store their numbers, this is useful for array shaping
        self.N_FLUXES = len(self.model.reactions)
        self.N_EX_RXNS = len(self.EX_RXNS)
        self.N_SYS_RXNS= len(self.SYS_RXNS)
        self.N_NEX_RXNS= len(self.NEX_RXNS)
        
        # Obtain initial upper and lower bounds of non exchange and non system reactions
        self.upper_bounds = OrderedDict()
        self.lower_bounds = OrderedDict()
        for rxn_id in self.NEX_RXNS:
            self.upper_bounds.update({rxn_id: self.model.reactions.get_by_id(rxn_id).upper_bound })
            self.lower_bounds.update({rxn_id: self.model.reactions.get_by_id(rxn_id).lower_bound })
                
        
        ######## Genotype ########
        # Store gene ids that are present in model.genes but also...
        self.GENES = np.array( [ gene.id for gene in self.model.genes ]  )
        
        #... other necessary genes for GRR that are not in model.genes
        if self.MODEL_NAME == 'ecoli_core': extra_genes.append('s0001')
        _ = [ self.GENES.append(gene) for gene in extra_genes ]
        
        # Create an ordered dict of gene dosages, initialized as a wt, i.e., g_i=1 for all i
        self.gene_dosage = OrderedDict()
        _ = [ self.gene_dosage.update({gene: 1.0}) for gene in self.GENES]
     
        # Store static variables
        self.N_GENES    = len(self.GENES)
        self.UPPER_CLIP  = 999
        self.LOWER_CLIP  = 0.1        
               
            
        ######## Gene reaction rules ########
        # Try to translate all gene reaction rules 
        #... as it contains lambda functions, this variable needs to be deleted before every parallelization.
        self.GRR = {}
        self.translate_all_gene_reaction_rules()

        
    
        if self.VERBOSE: print('The model has %d genes and %d reactions.' %(self.N_GENES, self.N_FLUXES) )
        if self.VERBOSE: print('... number of exchange reactions:\t%d' % self.N_EX_RXNS )
        if self.VERBOSE: print('... number of non-exchange reactions:\t%d' % self.N_NEX_RXNS )

    
    def slim_optimize(self):
        if "moma_old_objective" in self.model.solver.variables:
            _solution = self.model.optimize()
            if _solution.status=='optimal':
                return self.model.optimize().fluxes[self.BIOMASS_ID]
            else:
                return np.nan

        
        _slim_opt = self.model.slim_optimize( error_value = np.nan )
        if not np.isnan( _slim_opt ):
            return _slim_opt
        else:       
            _solution = self.model.optimize()
            if _solution.status=='infeasible':
                return np.nan  
    
    def optimize(self):
        return self.model.optimize()
    
    
    ###################################################################################
    ######################### COMPUTATION OF MAXIMAL BOUNDS ###########################
    ###################################################################################
    def save_bounds(self, filename, description=None):
        with open(filename, 'w+') as file:
            for rxn in self.NEX_RXNS:
                newline = '%s;%1.6f;%1.6f\r\n' % (rxn, self.upper_bounds[rxn], self.lower_bounds[rxn])
                file.write(newline)

            # The last line contains the number of samples and the lambda_E
            if description is not None:
                file.write('Description;%s' % description )
        print('Maxbounds exported as %s on %s.' % (filename, datetime.now() ) )
        return
        

    def old_load_bounds(self, filename):
        with open(filename,'r') as f:
            for _ in range(self.N_NEX_RXNS):
                data = f.readline().split('\n')[0].split(';')

                self.upper_bounds[data[0]] = float( data[1] )
                self.lower_bounds[data[0]] = float( data[2] )
            extra_data = f.readline()
        print('Maxbounds loaded from %s on %s.' % (filename, datetime.now() ) )
        return extra_data
    
    
    
    def load_bounds(self, filename):
        with open(filename,'r') as f:
             while True:
                    
                # Read next line of the csv file
                data = f.readline().split('\n')[0].split(';')
                
                #... if line is empty break loop
                if data[0]=='':                break
                
                #... if data[0] is recognized as a reaction ID, set bounds
                if data[0] in self.NEX_RXNS:
                    self.upper_bounds[data[0]] = float( data[1] )
                    self.lower_bounds[data[0]] = float( data[2] )
                    
                elif data[0] == 'Description':
                    self.bounds_description = ';'.join( data ) 
                else:
                    print('<W> Could not load bound for %s' % data[0])
                
        if self.VERBOSE:   print('Maxbounds loaded from %s on %s.' % (filename, datetime.now() ) )
    
    
    
    
    def reset_bounds(self):
        _ = [ self.__set_ubound(rxnid, +999) for rxnid in self.NEX_RXNS ]
        _ = [ self.__set_lbound(rxnid, -999) for rxnid in self.NEX_RXNS ]
   
        print('Maxbounds reset to +/- 999 on %s.' % (datetime.now() ) )
        return 
    
 
    def _compute_bounds_parallel(self, medium):

        from cobra.flux_analysis import pfba
        
        with self.model:
            self.model.medium = medium
            
            # First compute the unbounded solution
            if self._maxbounds_pfba:
                unbounded_sol = pfba(self.model)
            else:
                unbounded_sol = self.model.optimize()

            # Second, compute the bounded solution
            _nkd = int( self.N_NEX_RXNS*self._maxbounds_kd_frac)
            _rr  = np.random.randint( self.N_NEX_RXNS, size=(_nkd,))
            
            _ = [ self.__set_ubound(rxnid, +self._random_bound(0,100, 'linear')) for rxnid in self.NEX_RXNS[_rr] ]
            _ = [ self.__set_lbound(rxnid, -self._random_bound(0,100, 'linear')) for rxnid in self.NEX_RXNS[_rr] ]
            
            if self._maxbounds_pfba:
                bounded_sol = pfba(self.model)
            else:
                bounded_sol = self.model.optimize()

            
            # Third, extract both flux vectors
            _fluxes = np.zeros( ( self.N_NEX_RXNS,2) )
            _fluxes[:,0] = [ unbounded_sol.fluxes[rxnid] for rxnid in self.NEX_RXNS ]
            _fluxes[:,1] = [   bounded_sol.fluxes[rxnid] for rxnid in self.NEX_RXNS ]
            _mu        = unbounded_sol.fluxes[self.BIOMASS_ID]
            
            if bounded_sol.status == 'optimal':
                _maxbounds = [ self.__max_bound( F[0],F[1]) for F in _fluxes ]
            else:
                _maxbounds = [  unbounded_sol.fluxes[rxnid] for rxnid in self.NEX_RXNS ]
            
                
            return medium, _mu, _maxbounds

    
      

    def compute_bounds(self, media_list, kd_frac=0.10, pfba=True, nprocessors=None, chunksize=1):      
        ''' compute_bounds( media_list, kd_frac=0.10, pfba=True, nprocessors=None, chunksize=1)
        
        Computes the (p)fba solutions of the wild type and a mutant.
        From both flux solutions, only the maximum/minimum are returned.
        Computing the mutant helps to find the maxima of fluxes that are 
        only active when other reactions are turned off.
        
        The mutant has a fraction kd_frac with random bounds taken
        from a distribution U[0,100]. To modify these settings 
        you need to modify _compute_bounds_parallel() in the file of the class.
        '''
            
        # Compute as many samples as media in media_list
        samples = len(media_list)
        
        # The extra slot in _maxbounds_flx is for a full vector of 0s
        self._maxbounds_pfba = pfba
        self._maxbounds_flx  = np.zeros( ( samples + 1 , self.N_NEX_RXNS ) )
        self._maxbounds_bio  = np.zeros( ( samples  , ) )
        self._maxbounds_kd_frac = kd_frac
        self._maxbounds_media= []
        #... delete gene_reaction_rules as it makes Q non-parallelizable 
        self.GRR = {}

        
        with Pool( processes = nprocessors ) as pool:
                it = pool.imap_unordered( self._compute_bounds_parallel , media_list, chunksize=chunksize )
                icount = 0
                
                for results in tqdm( it, total=samples, leave=self.VERBOSE ):
                    self._maxbounds_media.append( results[0] )
                    self._maxbounds_bio[ icount ]   = results[1]
                    self._maxbounds_flx[ icount ,:] = results[2]
                    icount += 1
                    
        
        
        #==== Define global bounds ====#
        sleep(0.1)
        _ = [ self.upper_bounds.update( {rxn : np.max(self._maxbounds_flx[:,jj]) } ) for jj, rxn in enumerate(self.NEX_RXNS) ]
        _ = [ self.lower_bounds.update( {rxn : np.min(self._maxbounds_flx[:,jj]) } ) for jj, rxn in enumerate(self.NEX_RXNS) ]
        #print('There have been %d infeasible cases out of %d samples' % (_infeasible, samples) )
        
        # And recompute gene reaction rule translations (aka lambda functions)
        self.translate_all_gene_reaction_rules()

         
    def _random_bound(self, fmin, fmax, method):
                      
        # The maximum minimal bound is positive, non-null and smaller than fmax
        fmin = np.clip( fmin , 0.01, fmax )
        
        if method == 'linear':
            return (fmax - fmin) * np.random.rand() + fmin
        
        if method == 'log':        
            logmin = np.log10(fmin)
            logmax = np.log10(fmax)
            return 10** ( logmin + (logmax-logmin)*np.random.rand() )  
        

    def __max_bound(self,b1,b2): 
        #Both bounds share sign
        if b1*b2 > 0:
            return np.sign(b1)*np.nanmax(( abs(b1), abs(b2) ))
        
        #Bounds differ in sign
        if b1*b2 < 0:
            return np.nanmax(( abs(b1), abs(b2) ))
        
        #One of the bounds is null
        if b1 == 0:
            return b2
        
        if b2 == 0:
            return b1
            
            
    def __set_ubound(self, reaction_id , upper_bound):
        if self.__is_forwardible( reaction_id ):
            self.model.reactions.get_by_id( reaction_id ).upper_bound = upper_bound


    def __set_lbound(self, reaction_id , lower_bound):
        if self.__is_reversible(reaction_id) :
            self.model.reactions.get_by_id( reaction_id ).lower_bound = lower_bound


    def __is_reversible(self, reaction_id):
        return self.model.reactions.get_by_id( reaction_id).lower_bound < 0


    def __is_forwardible(self, reaction_id):
        return self.model.reactions.get_by_id( reaction_id).upper_bound > 0


    
    
    ##################################################################################
    ######################### GENE REACTION RULE TRANSLATION #########################
    ##################################################################################
    
    def translate_all_gene_reaction_rules(self):
        # Create empty dictionary with all reaction rules and their lambda functions
        self.GRR = {}
        
        # For every reactions...
        for rxn_id in self.NEX_RXNS:
            #... obtain the gene reaction rule raw string
            grr = self.model.reactions.get_by_id(rxn_id).gene_reaction_rule
            
            #... make sure it has the expected format
            grr = self._correct_reaction_rule(grr)
            
            #... build its mathematical interpretation
            temp_data = 'lambda self, : ' + self._grr_to_dosage( grr )
            
            #... save in the dictionary
            self.GRR.update( { rxn_id: eval(temp_data) } )
    
    
    def _correct_reaction_rule(self, reaction_rule):
        # If reaction rule is empty, return as is
        if reaction_rule == '':
            return reaction_rule
        
        # Add blanks between all types of parenthesis
        reaction_rule = reaction_rule.replace('(', ' ( ')
        reaction_rule = reaction_rule.replace(')', ' ) ')
        
        #... also remove all double, triple blanks ...
        while reaction_rule.find('  ') != -1:
            reaction_rule = reaction_rule.replace('  ', ' ')

        # If first character is a blank, also remove it
        if reaction_rule[0]==' ':
            reaction_rule = reaction_rule[1:]
        
        # If last character is a blank, also remove it
        if reaction_rule[-1]==' ':
            reaction_rule = reaction_rule[:-1]
            
        return reaction_rule
            
            
    def _grr_to_dosage(self, reaction_rule):
        # WARNING #
        # Reaction rules cannot be enclosed into "general" parenthesis!
        
        # If reaction rule is empty set it unbounded
        if reaction_rule == '':
            return '99999'
            
        # If it is not empty, go through all the depths:
        while self._compute_depth(reaction_rule)>0:
            
            # Find first subreaction
            lidx, ridx  = self._get_next_level(reaction_rule.split(' '))
            subreaction = reaction_rule[(lidx+2):(ridx-1)]   
            
            # Translate subreaction
            translated_subreaction = self._substitute_basic_rule( subreaction.split(' '))
            
            # Merge translation to original reaction
            reaction_rule = reaction_rule[:lidx] + translated_subreaction + reaction_rule[(ridx+1):]
        
        # Once all parenthesis are removed, depth==0, apply final substitution
        final_reaction = self._substitute_basic_rule( reaction_rule.split(' '))
        
        return final_reaction
    
    
    def _compute_depth(self, reaction):
        return np.sum([1 for elem in reaction.split(' ') if elem==')'])
    
    
    def _substitute_basic_rule(self, subreaction):
        elems_involved = [elem for elem in subreaction if elem != 'and' and elem!='or']
        logic_involved = np.array( [elem for elem in subreaction if elem=='and' or elem=='or'])

        if len(logic_involved)==0:
            return 'self.gene_dosage["' + subreaction[0] + '"]'

        elif  all( logic_involved == 'and'):
            q_rxn = 'min(('

        elif  all( logic_involved == 'or' ):
            q_rxn = 'sum(('
            
        else:
            print('<< E >> Rule unclear. Clarify with the use of parenthesis.')
            print(subreaction)
            return

        for elem in elems_involved:
            if elem[:3]=='min' or elem[:3]=='sum':
                q_rxn += elem+','
            else:
                q_rxn += 'self.gene_dosage["' + elem + '"],'

        return q_rxn[:-1]+'))'   
    
    
    def _get_next_level(self, reaction_rule): 
        #... find all left parenthesis
        temp_idx = np.array( [pos for pos, elem in enumerate(reaction_rule ) if elem == '('] )
        lidx = np.array( [sum([len(reaction_rule[jj])+1 for jj in range(temporal_idx)]) for temporal_idx in temp_idx] )


        #... find first right parenthesis
        temp_idx = np.array( [pos for pos, elem in enumerate(reaction_rule ) if elem == ')'] )
        ridx = np.array( [sum([len(reaction_rule[jj])+1 for jj in range(temporal_idx)]) for temporal_idx in temp_idx])
        ridx = ridx[0]
        
        #... among all left parenthesis, choose closest to the left of ridx
        lidx = lidx[ np.max( np.where( (lidx-ridx)<0)) ]
        
        return lidx, ridx
    
    ##################################################################################
    ########################### Individuals and populations ##########################
    ##################################################################################
    
    def __old_random_medium(self, noc = 10, minimal={}, lower_lim=0, upper_lim = 20, method='linear'):
        
        # If noc is None return current medium
        if noc is None:
            return self.model.medium

        #... else clip the value of noc between 0 and the number of exchange reactions
        noc = np.clip( noc, 0, self.N_EX_RXNS ) 
        
        # Create empty dict
        new_medium = {}

        # Randomly select "noc" number of components...
        components = np.random.permutation(self.EX_RXNS)[:noc]

        #... sample environment_randomization_fun
        _ = [new_medium.update( { icomp:self._random_bound( lower_lim, upper_lim, method) } ) for icomp in components ]
        
        # Then add minimal components
        _ = [ new_medium.update( { key : value } ) for key, value in minimal.items() ]
        
        return new_medium
    

    def random_medium(self, m , minimal={}, lower_lim=0, upper_lim = 20, method='linear'):
        
        # If noc is None return current medium
        if m is None or m<=0:
            return self.model.medium

        # Create empty dict
        new_medium = {}

        # Sample exponential distribution with lambda=m
        s_exp = np.random.exponential(scale=m, size=(self.N_EX_RXNS,))
        s_uni = np.random.rand(self.N_EX_RXNS)
        components = self.EX_RXNS[ np.nonzero(s_uni<s_exp)]



        #... sample environment_randomization_fun
        _ = [new_medium.update( { icomp:self._random_bound( lower_lim, upper_lim, method) } ) for icomp in components ]
        
        # Then add minimal components
        _ = [ new_medium.update( { key : value } ) for key, value in minimal.items() ]
        
        return new_medium


    


    

    def random_genotype(self, sigma, method, lower_lim=1e-12, upper_lim = 1 ):
        if method =='exponential':
            genotype = 1- np.random.exponential( scale=sigma[0] , size=(self.N_GENES,))

        elif method =='normal':
            genotype = sigma[0] + sigma[1]*np.random.randn( self.N_GENES )

        elif method == 'uniform':
            genotype = (sigma[1]-sigma[0])*np.random.rand( self.N_GENES ) + sigma[0]

        return np.clip( genotype  , lower_lim , upper_lim )


    def compute_individual(self, genotype = None, medium = None ):
        # Reset random seed for better randomizations (useful when parallel computing)
        #np.random.seed()
        
        with self.model:

            # Set random medium
            self.set_medium( medium )
            
            # Obtain media richness as the WT's growth rate
            #mu_wt = self.model.optimize().fluxes[self.BIOMASS_ID]
            mu_wt = self.slim_optimize()
            #... if infeasible, abort
            if mu_wt is None or np.isnan(mu_wt) or mu_wt==0:
                return None
            
            
            # Set specific genotype
            gen = self.set_genotype( genotype )
            self.apply_dosage()            
            
            # ... check feasibility
            solution = self.model.optimize()
            if solution.status != 'optimal':
                return None
           
        
            # Prepare output data
            gen      = [ self.gene_dosage[gene] for gene in self.GENES ]
            medium   = self.model.medium
            mu_mt    = solution.fluxes[self.BIOMASS_ID]
            flx_mt   = [ solution.fluxes[rxn_id] for rxn_id in self.NEX_RXNS ]

        return mu_mt, gen, flx_mt, medium, mu_wt

    
    
    def compute_population(self, genotypes, media, nprocessors=50, chunksize=1, timeout=20):
        from multiprocessing import TimeoutError

        samples     = genotypes.shape[0]
        _infeasible = 0
        
        # Preallocate memory for output data
        mu       = np.nan*np.zeros((samples,))
        J        = np.nan*np.zeros((samples, self.N_NEX_RXNS))
        Ri       = np.nan*np.zeros((samples,))
        
        if self.VERBOSE: print('Computing population of %d samples...' % samples)

        # Prepare parallel function
        global parallel_function
        def parallel_function(idx):
            return idx, self.compute_individual( genotype = genotypes[idx,:] ,  medium=media[idx] )
        
        # Prepare computations
        for _nth in ('2nd','3rd','4th','5th'):
            with self.model, Pool( processes = nprocessors) as pool, tqdm(total=samples, leave=self.VERBOSE) as pbar:           
                # This helps reduce the search space and give consistent results
                self.reset_dosage()
                self.apply_dosage()

                # Create iterator with parallel function
                it = pool.imap_unordered( parallel_function , range(samples), chunksize=chunksize )

                try:
                    for _ in range(samples):
                        pbar.update(1)
                        results = it.next(timeout=timeout)
                        idx     = results[0]
                        if results[1] is None:
                            _infeasible += 1
                            continue
                        else:
                            mu[idx] = results[1][0]
                            J[idx,:]= results[1][2]
                            Ri[idx] = results[1][4]

                    # When finished looping over the results, print number of infeasible cases...
                    if self.VERBOSE: print('Total infeasible from within %d' % _infeasible)
                        
                    #... and return output
                    return mu, genotypes, J, media, Ri

                # If a TimeoutError occurs, restart computation
                except TimeoutError:
                    print('< W > Parallel function timed out. Restarting for the %s time.' % _nth )
            
        # After trying for the 5th time unseccesfully, return nans
        mu[:]  = np.nan
        J[:]   = np.nan
        Ri[:]  = np.nan
        return mu, genotypes, J, media, Ri
    
    
    def clear_population(self, mu , MATRICES=[], minimal_gr=1e-2):      
        idx = np.isnan(mu) | (mu < minimal_gr)
        _mu = np.delete(mu, idx, axis=0)
        _M  = [ np.delete(jj, idx, axis=0) for jj in MATRICES ]
        if self.VERBOSE: print('\n <W> Clearing %d individuals from the population' % np.sum(idx) )
        return _mu, *_M
    
    
    '''
    #### OLD VERSION WITHOUT TIMEOUT ###
    def compute_population(self, genotypes, media, nprocessors=50, chunksize=1):
        samples     = genotypes.shape[0]
        _infeasible = 0
        
        # Preallocate memory for output data
        mu       = np.nan*np.zeros((samples,))
        J        = np.nan*np.zeros((samples, self.N_NEX_RXNS))
        Ri       = np.nan*np.zeros((samples,))
        

        # Prepare parallel function
        global parallel_function
        def parallel_function(idx):
            return idx, self.compute_individual( genotype = genotypes[idx,:] ,  medium=media[idx] )
        
        
        with self.model:
            # This helps reduce the search space and give consistent results
            self.reset_dosage()
            self.apply_dosage()
            
            # Open pool of workers and start parallelizing the parallel_function
            if self.VERBOSE: print('Computing population of %d samples...' % samples)
            with Pool( processes = nprocessors) as pool:
                it = pool.imap_unordered( parallel_function , range(samples), chunksize=chunksize )

                for results in tqdm(it, total=samples, leave=self.VERBOSE): 
                    idx = results[0]

                    # If result is infeasible, skip:
                    if results[1] is None:
                        _infeasible += 1
                        continue

                    #... or else store output data
                    mu[idx] = results[1][0]
                    J[idx,:]= results[1][2]
                    Ri[idx] = results[1][4]
        if self.VERBOSE: print('Total infeasible from within %d' % _infeasible)
        
        return mu, genotypes, J, media, Ri
    '''
    
    
    

    
    ##################################################################################
    ######################### ------------------------------ #########################
    ##################################################################################
    def set_dosage(self, gene_id, relative_dosage=1):
        self.gene_dosage[gene_id] = np.clip( relative_dosage, 1e-99, 1e3)
        return relative_dosage
    
    def reset_dosage(self):
        [self.set_dosage(gene, 1) for gene in self.GENES] 

    def apply_dosage(self):
        [ self.__set_ubound( rxn_id, np.clip( self.GRR[rxn_id](self) , 0, 1)*self.upper_bounds[rxn_id]) 
         for rxn_id in self.NEX_RXNS ]
        
        [ self.__set_lbound( rxn_id, np.clip( self.GRR[rxn_id](self) , 0, 1)*self.lower_bounds[rxn_id] ) 
         for rxn_id in self.NEX_RXNS ]
        
        
    def set_genotype(self, genotype):
        if genotype is None:
            genotype = np.ones((self.N_GENES,))
            
        _=[ self.set_dosage(gene, genotype[jj] ) for jj, gene in enumerate(self.GENES) ]
        return genotype
    
        
    def set_medium(self, medium=None):
        if medium is not None:
            self.model.medium = medium
        return self.model.medium
    
    
    
    
    
    
    
    
    

                
            
            
            
          
        
        
