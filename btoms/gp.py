import numpy as np
import btoms
import scipy
import copy
import ase


class GaussianProcess():
    """Gaussian Process Regression method.

    Parameters
    ----------
    kernel : Kernel object
        Kernel object from btoms.kernel
    """

    def __init__(self, kernel=None):
        if kernel is None:
            kernel = btoms.GRBF(0.1) + btoms.GWhiteNoise(1e-4, 'fixed')
        self.kernel = kernel
        self.optimized = False
        self.scaling = 1
        self.a = None
        self.L = None
        self.X = None
        self.y = None

    def fit(self, X, Y, optimize=False):
        """Fit an instance of the surrogate model.

        Parameters
        ----------
        X : ndarray (N, D)
            Observed atomic positions.
        Y : ndarray (N, D+1)
            Targets energies and forces.
        optimize : bool
            Whether to optimize the hyperparameters.
        """
        self.X = X.copy()

        Y = Y.copy()
        self.scaling = Y[:, 0].mean()
        Y[:, 0] -= self.scaling
        self.y = Y.flatten()

        self.optimized = optimize
        if self.optimized:
            init_theta = self.kernel.theta.copy()
            try:
                opt = scipy.optimize.minimize(
                    self.nlog_likelihood, init_theta,
                    bounds=self.kernel.bounds,
                    tol=1e-5, jac=True)
                self.kernel.theta = opt.x
            except(np.linalg.linalg.LinAlgError):
                self.kernel.theta = init_theta
                self.optimized = False

        if not self.optimized:
            K = self.kernel(X)
            self.L = scipy.linalg.cho_factor(K, lower=True)[0]
            self.a = scipy.linalg.cho_solve((self.L, True), self.y)

    def nlog_likelihood(self, theta, eval_gradient=True):
        """Return the negative of the log marginal likelihood.

        Parameters
        ----------
        theta : tuple (N,)
            Hyperparameters to be optimized.
        eval_gradient : bool
            Whether to evaluate the gradient of the kernel.

        Returns
        -------
        lml : float
            Negative log marginal likelihood.
        glml : ndarray (N,)
            Gradient of the negative log marginal likelihood.
        """
        self.kernel.theta = theta
        if eval_gradient:
            K, K_grad = self.kernel(self.X, eval_gradient=True)
        else:
            K = self.kernel(self.X)

        self.L = scipy.linalg.cho_factor(K, lower=True)[0]
        self.a = scipy.linalg.cho_solve((self.L, True), self.y)

        lml = -0.5 * np.dot(self.y, self.a)
        lml -= np.log(np.diag(self.L)).sum()
        lml -= self.X.shape[0] * 0.5 * np.log(2 * np.pi)

        if not eval_gradient:
            return -lml

        glml = np.outer(self.a, self.a)
        glml -= scipy.linalg.cho_solve(
            (self.L, True), np.eye(K.shape[0]))
        glml = 0.5 * np.einsum(
            'ijl,ijk->kl', glml[:, :, None], K_grad)

        return -lml, -glml

    def predict(self, Y, covariance=False):
        """Return a prediction of the currently trained model.
        Can only accept a single input.

        Parameters
        ----------
        Y : ndarray (D,)
            The position at which the prediction is computed.
        covariance : bool
           Whether to return the convenience matrix.

        Returns
        -------
        y : ndarray (D+1,)
            The predicted energy and forces
        covariance : ndarray (D+1, D+1)
            Covariance matrix, its diagonal is the variance.
        """
        k = np.hstack([self.kernel(X, Y) for X in self.X])
        y = np.dot(k, self.a)
        y += self.scaling

        if not covariance:
            return y

        assert Y.ndim == 1
        v = scipy.linalg.solve_triangular(self.L, k.T, lower=True)
        covariance = self.kernel(Y)
        covariance -= np.tensordot(v, v, axes=(0, 0))

        return y, covariance


class GPCalculator(ase.calculators.calculator.Calculator, GaussianProcess):
    """Gaussian Process calculator for ASE.

    Parameters
    ----------
    kernel : Kernel object
        Kernel object from btoms.kernel
    eps : float
        Incremental step for calculation of forces through finite difference.
    """

    implemented_properties = ['energy', 'forces']

    def __init__(self, kernel=None, eps=1e-8):
        GaussianProcess.__init__(self, kernel)
        ase.calculators.calculator.Calculator.__init__(self)
        self.eps = eps

    def calculate(self, atoms=None, properties=['energy', 'forces'],
                  system_changes=None):

        if atoms is not None:
            self.atoms = atoms.copy()

        cons = atoms.constraints[0].todict()['kwargs']['indices']
        mask = ~np.in1d(range(len(atoms)), cons)

        # Energy
        X = atoms.positions[mask].flatten()
        energy, covariance = self.predict(X, True)

        # Forces by finite difference
        D = X.shape[0]
        X = np.tile(X, (D, 1))
        eps_diagonal = np.eye(D) * self.eps
        Xp, Xn = X + eps_diagonal, X - eps_diagonal

        grad = self.predict(np.vstack([Xp, Xn]))
        diff = (grad[:D] - grad[D:2*D]) / (2 * self.eps)

        forces = np.zeros((len(atoms), 3))
        forces[mask] = -diff.reshape(-1, 3)

        self.results = {
            'energy': energy,
            'forces': forces,
            'uncertainty': np.sqrt(covariance[0][0])
        }

    def fit_to_images(self, images, optimize=False):
        """Train a Gaussian process using a provided set of
        Atoms objects.

        Parameters
        ----------
        images : list of Atoms objects
            Images to use as training data. All images must have
            evaluated energies and forces.
        optimize : bool
            Whether to optimize the hyperparameters.
        """
        if not isinstance(images, list):
            images = [images]

        cons = images[0].constraints[0].todict()['kwargs']['indices']
        mask = ~np.in1d(range(len(images[0])), cons)

        D = 3 * len(np.where(mask)[0])
        N = len(images)

        X, Y = np.empty((N, D)), np.empty((N, D + 1))
        for i, atoms in enumerate(images):
            X[i] = atoms.positions[mask].flatten()
            Y[i, 0] = atoms.get_potential_energy()
            Y[i, 1:] = -atoms.get_forces()[mask].flatten()

        self.fit(X, Y, optimize)

    def get_uncertainty(self):
        """Return the uncertainty of the Gaussian Process."""
        return self.results['uncertainty']

    def copy(self):
        """Create a copy of the calculator."""
        calc = copy.deepcopy(self)

        return calc
