import numpy as np
from abc import ABCMeta, abstractmethod, abstractproperty
from collections import namedtuple
from scipy.stats import norm

class Kappa:
    def __init__(self, kappa, decay=0, scheduler='exponential'):
        self.kappa = kappa 
        self.kappa_ini = self.kappa
        self.decay = decay
        self.counter = 0
        self.scheduler = scheduler 
    def get_kappa(self):
        rval = self.kappa
        self.counter += 1
        if self.scheduler == 'exponential':
            self.kappa = self.kappa_ini * (np.exp(-self.counter * self.decay))
        if self.scheduler == 'linear':
            if self.kappa >= 0:
                self.kappa = self.kappa_ini - self.counter * self.decay 
            else: 
                self.kappa = self.kappa
        return rval 
    

class InconsistencyError(Exception):
    pass 

class AcquisitionFunction(metaclass=ABCMeta):
    """Baseclass for all the acquisition functions."""

    @abstractproperty 
    def params(self):
        yield None 

    def get_params(self):
        return self.params

    @abstractmethod
    def __call__(self, mu, sigma, ybest):
        """Evaluate the acquisition function."""

    @abstractmethod
    def __repr__(self):
        """Rule to print out a human readable description of the acquisition
        function."""

    @abstractproperty
    def name(self):
        yield None

    def __add__(self, other):
        if not isinstance(other, AcquisitionFunction):
            raise InconsistencyError('You can only add an AcquisitionFunction')
        return AFSum(self, other)

    def __mul__(self, other):
        if not isinstance(other, AcquisitionFunction):
            raise InconsistencyError('You can only'
                                     ' multiply an AcquisitionFunction')
        return AFMul(self, other)
    
    def __eq__(self, other):
        if self.name == other.name and self.params == other.params:
            return True
        else:
            return False

class AFOperator(AcquisitionFunction):
    """Base class for all AF operator."""
    def __init__(self, af1, af2):
        self.af1 = af1  
        self.af2 = af2

    def get_params(self):
        params = dict(af1=self.af1, af2=self.af2)
        params.update(('af1__' + af, val) 
                for af, val in self.af1.params.items())
        params.update(('af2__' + af, val) 
                for af, val in self.af2.params.items())
        return params 

class AFSum(AFOperator):
    pass

class AFMul(AFOperator):
    pass

class PI(AcquisitionFunction):
    """Probability of improvement acquisition function. The function is 
    given by:

    PI(x) = Phi((mu(x) - f(x+) - kappa)/ sigma(x))
    Phi is the cumulative distribution function (cdf). 
 
    Parameters
    ----------
    kappa : exploration/exploitation tradeoff value. 
            0 means purely exploitative while more positive value means 
            explorative.
    eval_grad : bool
        Wheather to use force and force uncertainty in the 
        acquisition function computation."""
    
    def __init__(self, kappa, eval_grad=False):
        self.kappa = kappa
        self.eval_grad = eval_grad 
        self.name = 'PI'
        self.params = dict(kappa=self.kappa, eval_grad=self.eval_grad)

    def __repr__(self):
        return "{}({})".format(self.name, self.params)

    def name(self):
        return self.name

    def params(self):
        return self.params

    def __call__(self, y_best=None, m=None, s=None, dmdx=None, dsdx=None):
        """Return the evaluated acquisition function values.
        Parameters
        ----------
        y_best : the best y value found until now.
        m : mean values or energies returned by the Gaussian predictor.
        s : standard deviation or uncertainty in energy prediction returned 
        by the Gaussian predictor.
        dmdx : force values returned by the Gaussian predictor.
        dsdx : standard deviation or uncertainty in force prediction returned 
        by the Gaussian predictor.
        """
        self.fvalue = norm.cdf((m-y_best-self.kappa.get_kappa())/s) 
        if self.eval_grad:
            raise NotImplementedError('Method not yet implemented.')
        return self.fvalue

class EI:
    """Expected improvement acquisition function. The function is given by:

    EI(x) = (mu(x) - f(x+)) * Phi(Z) + sigma(x) * phi(Z)
    where Z = (mu(x) - f(x+)/sigma(x))
    Phi and phi are cdf and pdf, respectively. 
    
    Parameters
    ----------
    kappa : exploration/exploitation tradeoff value. 
            0 means purely exploitative while more positive value means 
            explorative.
    eval_grad : bool
        Wheather to use force and force uncertainty in the 
        acquisition function computation."""
    
    def __init__(self, kappa, eval_grad=False):
        self.kappa = kappa
        self.eval_grad = eval_grad 
        self.name = 'EI'
        self.params = dict(kappa=self.kappa, eval_grad=self.eval_grad)

    def __repr__(self):
        return "{}({})".format(self.name, self.params)

    def name(self):
        return self.name

    def params(self):
        return self.params

    def __call__(self, y_best=None, m=None, s=None, dmdx=None, dsdx=None):
        """Return the evaluated acquisition function values.
        Parameters
        ----------
        y_best : the best y value found until now.
        m : mean values or energies returned by the Gaussian predictor.
        s : standard deviation or uncertainty in energy prediction returned 
        by the Gaussian predictor.
        dmdx : force values returned by the Gaussian predictor.
        dsdx : standard deviation or uncertainty in force prediction returned 
        by the Gaussian predictor.
        """
        kappa = self.kappa.get_kappa()
        Phi = norm.cdf((m-y_best-kappa)/s) 
        phi = norm.pdf((m-y_best-kappa)/s) 
        self.fvalue = (m-y_best) * Phi + s * phi
        if self.eval_grad:
            raise NotImplementedError('Method not yet implemented.')
        return self.fvalue
    
class ES:
    pass

class UCB:
    """Upper confidence bound acquisition function. The function is given by:

    UCB(x) = mu(x) + kappa * sigma(x)
    
    Parameters
    ----------
    kappa : exploration/exploitation tradeoff value. 
            0 means purely exploitative while more positive value means 
            explorative.
    eval_grad : bool
        Wheather to use force and force uncertainty in the 
        acquisition function computation."""
    
    def __init__(self, kappa, eval_grad=False):
        self.kappa = kappa
        self.eval_grad = eval_grad 
        self.name = 'UCB'
        self.params = dict(kappa=self.kappa, eval_grad=self.eval_grad)

    def __repr__(self):
        return "{}({})".format(self.name, self.params)

    def name(self):
        return self.name

    def params(self):
        return self.params

    def __call__(self, y_best=None, m=None, s=None, dmdx=None, dsdx=None):
        """Return the evaluated acquisition function values.
        Parameters
        ----------
        y_best : the best y value found until now.
        m : mean values or energies returned by the Gaussian predictor.
        s : standard deviation or uncertainty in energy prediction returned 
        by the Gaussian predictor.
        dmdx : force values returned by the Gaussian predictor.
        dsdx : standard deviation or uncertainty in force prediction returned 
        by the Gaussian predictor.
        """
        self.fvalue = m + self.kappa.get_kappa() * s
        if self.eval_grad:
            raise NotImplementedError('Method not yet implemented.')
        return self.fvalue
    
class LP:
    pass

