import numpy as np
import ase.neb
import btoms
import copy


class Converged(Exception):
    pass


class NEB():
    """Bayesian optimization for nudged elastic band.

    Parameters
    ----------
    trajectory : list of Atoms objects
        The NEB trajectory to optimize.
    calculator : Calculator object
        ASE calculator.
    kernel : Kernel object
        Kernel to be used in the GP calculator btoms.kernel.
    targets : list of Atoms objects | str
        Previously trained atoms objects to use as training data.
    """

    def __init__(self, trajectory, calculator=None, kernel=None, targets=None):
        self.images = trajectory
        self.gaussian = btoms.GPCalculator(kernel)
        self.calculator = calculator
        self.iteration = 0

        if targets is None:
            self.targets = [trajectory[0], trajectory[-1]]
        elif isinstance(targets, str):
            self.targets = ase.io.read(targets)
        ase.io.write('targets.json', self.targets)

        tmp = np.empty(len(self.images) - 2)
        self.data = {'energy': tmp.copy(), 'sigma': tmp.copy()}

    def run(self, fmax=0.05, umax=0.05, steps=200, gpsteps=1000):
        """Executing run will start the NEB optimization process.

        Parameters
        ----------
        fmax : float
            Maximum absolute force component to consider saddle point
            converged (eV/Angs).
        umax : float | tuple of float (2,)
            For all images, the maximum allowed GP standard deviation
            be considered converged (eV). Also defines the maximum allowed
            uncertainty before terminating a NEB relaxation early.
        steps : int
            Maximum number of calculator evaluations.
        gpsteps : int
            Maximum number of iterations for GP directed NEB relaxation.
        """
        if len(self.targets) == 2:
            target = self.images[len(self.images) // 2]
            atoms = self.evaluate_atoms(target)
            self.targets += [atoms]

        initial_images = self.images

        saddle_found = False
        for i in range(steps):
            self.gaussian.fit_to_images(self.targets, optimize=True)

            self.images = copy.deepcopy(initial_images)
            for image in self.images[1:-1]:
                image.set_calculator(self.gaussian.copy())

            # Perform NEB on surrogate model
            try:
                neb = ase.neb.NEB(
                    self.images,
                    climb=True,
                    method='improvedtangent')
                opt = ase.optimize.MDMin(neb, dt=0.05, logfile=None)
                opt.attach(self.check_convergence, 1, umax)
                opt.run(fmax=fmax * 0.8, steps=gpsteps)
            except(Converged):
                pass
            ase.io.write('neb.json', self.images)

            # Acquisition function
            uncertainty_converged = self.data['sigma'].max() < umax
            target = self.data['sigma']
            if uncertainty_converged and not saddle_found:
                target += self.data['energy']
            target = np.argmax(target) + 1

            # Evaluate a new target
            atoms = self.images[target]
            atoms = self.evaluate_atoms(atoms)
            self.targets += [atoms]

            force_converged = np.abs(atoms.get_forces()).max() < fmax
            if force_converged:
                saddle_found = True

                if uncertainty_converged:
                    return True

    def evaluate_atoms(self, atoms):
        """Calculate the energy of a given structure with the
        provided ASE calculator.
        """
        atoms = atoms.copy()
        atoms.set_calculator(self.calculator)
        atoms.get_potential_energy()
        self.targets += [atoms]

        self.iteration += 1
        print('Evaluation {}'.format(self.iteration))
        ase.io.write('targets.json', atoms, append=True)

        return atoms

    def check_convergence(self, umax):
        """Return a convergence criteria for the surrogate model."""
        for i, atoms in enumerate(self.images[1:-1]):
            self.data['energy'][i] = atoms.get_potential_energy()
            self.data['sigma'][i] = atoms.calc.get_uncertainty()

        sigma = self.data['sigma']

        if sigma.max() > 2 * umax:
            raise Converged
