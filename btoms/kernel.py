import sklearn.gaussian_process
import numpy as np


class GRBF(sklearn.gaussian_process.kernels.RBF):
    """Radial-basis function kernel (aka squared-exponential kernel).
    with gradients.

    Parameters
    ----------
    length_scale : float
        The length scale of the kernel.
    length_scale_bounds : tuple of floats (2,)
        The constant component of the kernel.
    """

    def __init__(self, length_scale=1.0, length_scale_bounds=(1e-5, 1e5)):
        self.length_scale = length_scale
        self.length_scale_bounds = length_scale_bounds

        if self.anisotropic:
            raise ValueError('Only isotropic length scale supported.')

    def __call__(self, X, Y=None, eval_gradient=False):
        """Return the kernel k(X, Y) and optionally its gradient.

        Parameters
        ----------
        X : ndarray (N, D)
            Left argument of the returned kernel k(X, Y)
        Y : ndarray (N, D) | None
            Right argument of the returned kernel k(X, Y). If None, k(X, X)
            if evaluated instead.
        eval_gradient : bool
            Determines whether the gradient with respect to the kernel
            hyperparameter is determined.

        Returns
        -------
        K : ndarray (D, D)
            Kernel k(X, Y)
        K_gradient : array (opt.), shape (D, D, n_dims)
            The gradient of the kernel k(X, X) with respect to the
            hyperparameter of the kernel. Only returned when eval_gradient
            is True.
        """
        X = np.atleast_2d(X)
        self.D = X.shape[1] + 1
        if Y is not None and eval_gradient:
            raise ValueError("Gradient can only be evaluated when Y is None.")

        if Y is None:
            K = self.build_kernel(X, X, self.Kgx)
            if not eval_gradient:
                return K

            K_gradient = self.build_kernel(X, X, self.Kgl)[:, :, None]
            return K, K_gradient

        Y = np.atleast_2d(Y)
        K = np.ones((Y.shape[0], self.D))
        for i, y in enumerate(Y):
            K[i, 1:] = (y - X) / self.length_scale**2
            K[i] *= self.K(y, X)

        return K

    def K(self, X, Y):
        """Radial Basis Function."""
        y = np.exp(-0.5 * np.sum(((X - Y) / self.length_scale)**2))
        return y

    def Kgx(self, X, Y):
        """Build the kernel with its gradient components.
        This gradient is taken with respect to X and Y and includes
        the Hessian as well. TODO: include citation.
        """
        ls = self.length_scale
        K = np.eye(self.D)
        dx = X - Y

        # Jacobian
        K[0, 1:] = dx / ls**2
        K[1:, 0] = -K[0, 1:]

        # Hessian
        K[1:, 1:] -= np.outer(dx, dx) / ls**2
        K[1:, 1:] /= ls**2
        K *= self.K(X, Y)

        return K

    def Kgl(self, X, Y):
        """Build the kernel with its gradient components.
        This gradient is taken with respect to the length scale
        from each component of the X and Y gradient components.
        """
        ls = self.length_scale
        K = np.empty((self.D, self.D))
        dx = X - Y

        kl = np.dot(dx, dx) / ls**3
        sd = np.sum(dx**2) / ls**2

        pre = 1 - 0.5*sd
        J = 2 * pre * dx / ls**3

        P = np.outer(dx, dx) / ls**2
        H = -2 * (pre * (np.eye(X.shape[0]) - P) - P) / ls**3

        K[0][0] = kl
        K[1:, 0] = J
        K[0, 1:] = -K[1:, 0]
        K[1:, 1:] = H
        K *= self.K(X, Y)

        return K

    def build_kernel(self, X, Y, kernel):
        """Return a kernel comprised of multiple entries."""
        N = X.shape[0]
        K = np.eye(N * self.D)

        for i in range(N):
            b0 = slice(i * self.D, (i+1) * self.D)
            K[b0, b0] = kernel(X[i], Y[i])

            for j in range(i+1, N):
                b1 = slice(j * self.D, (j+1) * self.D)

                k = kernel(X[i], Y[j])
                K[b0, b1], K[b1, b0] = k, k.T

        return K


class GConstant(sklearn.gaussian_process.kernels.ConstantKernel):
    """Constant kernel adapted for included gradient.

    Parameters
    ----------
    constant_value : float
        The constant value which defines the covariance.
    constant_value_bounds: tuple of floats (2,)
        The lower and upper bound on constant_value.
    """

    def __init__(self, constant_value=1.0, constant_value_bounds=(1e-5, 1e5)):
        self.constant_value = constant_value
        self.constant_value_bounds = constant_value_bounds

    def __call__(self, X, Y=None, eval_gradient=False):
        """Return the kernel k(X, Y) and optionally its gradient.
        Corrected for the dimensionality of the gradient contribution,
        the shape is D = (n_samples_X * n_features) + 1

        Parameters
        ----------
        X : ndarray (N, D)
            Left argument of the returned kernel k(X, Y)
        Y : ndarray (N, D) | None
            Right argument of the returned kernel k(X, Y). If None, k(X, X)
            if evaluated instead.
        eval_gradient : bool
            Determines whether the gradient with respect to the kernel
            hyperparameter is determined.

        Returns
        -------
        K : ndarray (D, D)
            Kernel k(X, Y)
        K_gradient : array (opt.), shape (D, D, n_dims)
            The gradient of the kernel k(X, X) with respect to the
            hyperparameter of the kernel. Only returned when eval_gradient
            is True.
        """
        X = np.atleast_2d(X)
        if Y is not None and eval_gradient:
            raise ValueError("Gradient can only be evaluated when Y is None.")

        D = X.shape[0] * (X.shape[1] + 1)
        if Y is None:
            K = np.full((D, D), self.constant_value)
            if not eval_gradient:
                return K

            elif not self.hyperparameter_constant_value.fixed:
                K_gradient = K[:, :, None]
            else:
                K_gradient = np.empty((D, D, 0))

            return K, K_gradient
        else:
            return np.full(D, self.constant_value)


class GWhiteNoise(sklearn.gaussian_process.kernels.WhiteKernel):
    """White noise kernel adapted for gradients.

    Parameters
    ----------
    noise_level : float
        Parameter controlling the noise level.
    noise_level_bounds : tuple of floats (2,)
        The lower and upper bound on noise_level.
    """

    def __init__(self, noise_level=1.0, noise_level_bounds=(1e-5, 1e5)):
        self.noise_level = noise_level
        self.noise_level_bounds = noise_level_bounds

    def __call__(self, X, Y=None, eval_gradient=False):
        """Return the kernel k(X, Y) and optionally its gradient.

        Parameters
        ----------
        X : ndarray (N, D)
            Left argument of the returned kernel k(X, Y)
        Y : ndarray (N, D) | None
            Right argument of the returned kernel k(X, Y). If None, k(X, X)
            if evaluated instead.
        eval_gradient : bool
            Determines whether the gradient with respect to the kernel
            hyperparameter is determined.

        Returns
        -------
        K : ndarray (D, D)
            Kernel k(X, Y)
        K_gradient : array (opt.), shape (D, D, n_dims)
            The gradient of the kernel k(X, X) with respect to the
            hyperparameter of the kernel. Only returned when eval_gradient
            is True.
        """
        X = np.atleast_2d(X)
        if Y is not None and eval_gradient:
            raise ValueError("Gradient can only be evaluated when Y is None.")

        D = X.shape[0] * (X.shape[1] + 1)
        if Y is None:
            K = self.noise_level * np.eye(D)
            if not eval_gradient:
                return K

            elif not self.hyperparameter_noise_level.fixed:
                K_gradient = K[:, :, None]
            else:
                K_gradient = np.empty((D, D, 0))

            return K, K_gradient
        else:
            return np.zeros(D)
