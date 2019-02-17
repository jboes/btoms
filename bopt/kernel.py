from autograd import jacobian
import autograd.numpy as np
import sklearn.gaussian_process


class GRBF(sklearn.gaussian_process.kernels.RBF):
    """Radial-basis function kernel (aka squared-exponential kernel).
    with gradients.

    Parameters
    -----------
    length_scale : float or array with shape (n_features,), default: 1.0
        The length scale of the kernel. If a float, an isotropic kernel is
        used. If an array, an anisotropic kernel is used where each dimension
        of l defines the length-scale of the respective feature dimension.

    length_scale_bounds : pair of floats >= 0, default: (1e-5, 1e5)
        The lower and upper bound on length_scale

    """
    def __init__(self, length_scale=1.0, length_scale_bounds=(1e-5, 1e5)):
        self.length_scale = length_scale
        self.length_scale_bounds = length_scale_bounds

        # Define needed derivatives with respect to the input.
        self.J = jacobian(self.K)
        self.H = jacobian(self.J, 1)

        # Also, need derivatives with respect to the length scale
        self.Kl = jacobian(self.K, 2)
        self.Jl = jacobian(self.J, 2)
        self.Hl = jacobian(self.H, 2)

        if self.anisotropic:
            raise ValueError('Only isotropic length scale supported.')

    def __call__(self, X, Y=None, eval_gradient=False):
        """Return the kernel k(X, Y) and optionally its gradient.

        Parameters
        ----------
        X : array, shape (n_samples_X, n_features)
            Left argument of the returned kernel k(X, Y)

        Y : array, shape (n_samples_Y, n_features), (optional, default=None)
            Right argument of the returned kernel k(X, Y). If None, k(X, X)
            if evaluated instead.

        eval_gradient : bool (optional, default=False)
            Determines whether the gradient with respect to the kernel
            hyperparameter is determined. Only supported when Y is None.

        Returns
        -------
        K : array, shape (D, D)
            Kernel k(X, Y)

        K_gradient : array (opt.), shape (D, D, n_dims)
            The gradient of the kernel k(X, X) with respect to the
            hyperparameter of the kernel. Only returned when eval_gradient
            is True.
        """
        X = np.atleast_2d(X)
        if Y is None:
            Y = X

        n = X.shape[0]
        self.D = X.shape[1] + 1

        # Build the kernel.
        K = np.eye(n * self.D)
        for i in range(0, n):
            b0 = slice(i * self.D, (i+1) * self.D)
            K[b0, b0] = self.Kgx(X[i], Y[i], self.length_scale)

            for j in range(i+1, n):
                b1 = slice(j * self.D, (j+1) * self.D)

                k = self.Kgx(X[i], Y[j], self.length_scale)
                K[b0, b1], K[b1, b0] = k, k.T

        if not eval_gradient:
            return K

        # Build the gradient of the kernel.
        K_gradient = np.eye(n * self.D)
        for i in range(0, n):
            b0 = slice(i * self.D, (i+1) * self.D)
            K_gradient[b0, b0] = self.Kgl(X[i], Y[i], self.length_scale)

            for j in range(i+1, n):
                b1 = slice(j * self.D, (j+1) * self.D)

                k = self.Kgl(X[i], Y[j], self.length_scale)
                K_gradient[b0, b1], K_gradient[b1, b0] = k, k.T

        return K, K_gradient[:, :, None]

    def K(self, X, Y, ls):
        """Radial Basis Function."""
        y = np.exp(-0.5 * np.sum(((X - Y) / ls)**2))
        return y

    def Kgx(self, X, Y, ls):
        """Build the kernel with its gradient components.
        This gradient is taken with respect to X and Y and includes
        the Hessian as well. TODO: include citation.
        """
        K = np.empty((self.D, self.D))

        K[0][0] = self.K(X, Y, ls)
        K[1:, 0] = self.J(X, Y, ls)
        K[0, 1:] = -K[1:, 0]
        K[1:, 1:] = self.H(X, Y, ls)

        return K

    def Kgl(self, X, Y, ls):
        """Build the kernel with its gradient components.
        This gradient is taken with respect to the length scale
        from each component of the X and Y gradient components.
        """
        K = np.empty((self.D, self.D))

        K[0][0] = self.Kl(X, Y, ls)
        K[1:, 0] = self.Jl(X, Y, ls)
        K[0, 1:] = -K[1:, 0]
        K[1:, 1:] = self.Hl(X, Y, ls)

        return K


class GConstantKernel(sklearn.gaussian_process.kernels.ConstantKernel):
    """Constant kernel adapted for included gradient.

    Can be used as part of a product-kernel where it scales the magnitude of
    the other factor (kernel) or as part of a sum-kernel, where it modifies
    the mean of the Gaussian process.

    k(x_1, x_2) = constant_value for all x_1, x_2

    Parameters
    ----------
    constant_value : float, default: 1.0
        The constant value which defines the covariance:
        k(x_1, x_2) = constant_value

    constant_value_bounds : pair of floats >= 0, default: (1e-5, 1e5)
        The lower and upper bound on constant_value

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
        X : array, shape (n_samples_X, n_features)
            Left argument of the returned kernel k(X, Y)

        Y : array, shape (n_samples_X, n_features), (optional, default=None)
            Right argument of the returned kernel k(X, Y). If None, k(X, X)
            if evaluated instead.

        eval_gradient : bool (optional, default=False)
            Determines whether the gradient with respect to the kernel
            hyperparameter is determined. Only supported when Y is None.

        Returns
        -------
        K : array, shape (D, D)
            Kernel k(X, Y)

        K_gradient : array (opt.), shape (D, D n_dims)
            The gradient of the kernel k(X, X) with respect to the
            hyperparameter of the kernel. Only returned when eval_gradient
            is True.
        """
        X = np.atleast_2d(X)
        if Y is None:
            Y = X
        elif eval_gradient:
            raise ValueError("Gradient can only be evaluated when Y is None.")

        D = X.shape[0] * (X.shape[1] + 1)
        K = np.full((D, D), self.constant_value)
        if eval_gradient:
            if not self.hyperparameter_constant_value.fixed:
                K_gradient = np.full((D, D, 1), self.constant_value)
            else:
                K_gradient = np.empty((D, D, 0))
            return K, K_gradient
        else:
            return K

    def diag(self, X):
        """Returns the diagonal of the kernel.

        The result of this method is identical to np.diag(self(X)); however,
        it can be evaluated more efficiently since only the diagonal is
        evaluated.

        Corrected for the dimensionality of the gradient contribution,
        the shape is D = (n_samples_X * n_features) + 1

        Parameters
        ----------
        X : array, shape (n_samples_X, n_features)
            Left argument of the returned kernel k(X, Y)

        Returns
        -------
        K_diag : array, shape (D,)
            Diagonal of kernel k(X, X)
        """
        D = X.shape[0] * (X.shape[1] + 1)
        return np.full(D, self.constant_value)


class GWhiteKernel(sklearn.gaussian_process.kernels.WhiteKernel):
    """White kernel adapted for gradients.

    The main use-case of this kernel is as part of a sum-kernel where it
    explains the noise-component of the signal. Tuning its parameter
    corresponds to estimating the noise-level.

    k(x_1, x_2) = noise_level if x_1 == x_2 else 0

    Parameters
    ----------
    noise_level : float, default: 1.0
        Parameter controlling the noise level

    noise_level_bounds : pair of floats >= 0, default: (1e-5, 1e5)
        The lower and upper bound on noise_level

    """
    def __init__(self, noise_level=1.0, noise_level_bounds=(1e-5, 1e5)):
        self.noise_level = noise_level
        self.noise_level_bounds = noise_level_bounds

    def __call__(self, X, Y=None, eval_gradient=False):
        """Return the kernel k(X, Y) and optionally its gradient.

        Parameters
        ----------
        X : array, shape (n_samples_X, n_features)
            Left argument of the returned kernel k(X, Y)

        Y : array, shape (n_samples_Y, n_features), (optional, default=None)
            Right argument of the returned kernel k(X, Y). If None, k(X, X)
            if evaluated instead.

        eval_gradient : bool (optional, default=False)
            Determines whether the gradient with respect to the kernel
            hyperparameter is determined. Only supported when Y is None.

        Returns
        -------
        K : array, shape (n_samples_X, n_samples_Y)
            Kernel k(X, Y)

        K_gradient : array (opt.), shape (n_samples_X, n_samples_X, n_dims)
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
            if eval_gradient:
                if not self.hyperparameter_noise_level.fixed:
                    return K, self.noise_level * np.eye(D)[:, :, None]
                else:
                    return K, np.empty((D, D, 0))
            else:
                return K
        else:
            return np.zeros((D, D))

    def diag(self, X):
        """Returns the diagonal of the kernel k(X, X).

        The result of this method is identical to np.diag(self(X)); however,
        it can be evaluated more efficiently since only the diagonal is
        evaluated.

        Parameters
        ----------
        X : array, shape (n_samples_X, n_features)
            Left argument of the returned kernel k(X, Y)

        Returns
        -------
        K_diag : array, shape (n_samples_X,)
            Diagonal of kernel k(X, X)
        """
        D = X.shape[0] * (X.shape[1] + 1)
        return np.full(D, self.noise_level)
