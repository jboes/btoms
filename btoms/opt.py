import numpy as np
import ase.neb
import btoms


class Converged(Exception):
    pass


class NEB():
    """Bayesian optimization for nudged elastic band.

    Parameters
    ----------
    trajectory: list of Atoms objects
        Initial end-point of the NEB path or Atoms object.
    calculator: Calculator object
        ASE calculator as implemented in ASE.
        See https://wiki.fysik.dtu.dk/ase/ase/calculators/calculators.html
    """

    def __init__(self, trajectory, calculator=None):
        self.images = trajectory
        self.targets = [trajectory[0], trajectory[-1]]
        self.gaussian = btoms.GPCalculator()
        self.calculator = calculator
        self.iteration = 0

        tmp = np.empty(len(self.images) - 2)
        self.data = {'energy': tmp.copy(), 'sigma': tmp.copy()}

    def run(self, fmax=0.05, umax=0.05, steps=200, gpsteps=1000):
        """Executing run will start the NEB optimization process.

        Parameters
        ----------
        fmax : float
            Convergence criteria (in eV/Angs).
        umax: float
            Maximum uncertainty for convergence (in eV).
        steps : int
            Maximum number of evaluations on the calculator.
        gpsteps : int
            Maximum number of iterations for the surrogate model.
        """
        if len(self.targets) == 2:
            target = self.images[len(self.images) // 2]
            atoms = self.evaluate_atoms(target, False)
            self.targets += [atoms]

        saddle_found = False
        for i in range(steps):
            self.gaussian.fit_to_images(self.targets, optimize=True)

            for image in self.images[1:-1]:
                image.set_calculator(self.gaussian.copy())

            # Perform NEB on surrogate model
            try:
                neb = ase.neb.NEB(
                    self.images,
                    climb=True,
                    method='improvedtangent')
                opt = ase.optimize.MDMin(neb, dt=0.05, logfile=None)
                opt.attach(self.check_convergence)
                opt.run(fmax=fmax, steps=gpsteps)
            except(Converged):
                pass

            # Acquisition function
            uncertainty_converged = self.data['sigma'].max() < umax
            target = self.data['sigma']
            if uncertainty_converged and not saddle_found:
                target += self.data['energy']
            target = np.argmax(target) + 1

            # Evaluate a new target
            atoms = self.images[target]
            atoms = self.evaluate_atoms(atoms, append=True)
            self.targets += [atoms]

            force_converged = np.abs(atoms.get_forces()).max() < fmax
            if force_converged:
                saddle_found = True

                if uncertainty_converged:
                    ase.io.write('neb.json', self.images)
                    return True

    def evaluate_atoms(self, atoms, append=True):
        """Calculate the energy of a given structure with the
        provided ASE calculator.
        """
        atoms = atoms.copy()
        atoms.set_calculator(self.calculator)
        atoms.get_potential_energy()
        self.targets += [atoms]

        self.iteration += 1
        print('Evaluation {}'.format(self.iteration))

        if append is not None:
            ase.io.write('evaluated.json', atoms, append=append)
            ase.io.write('paths.json', self.images, append=append)

        return atoms

    def check_convergence(self):
        """Return a convergence criteria for the surrogate model."""
        for i, atoms in enumerate(self.images[1:-1]):
            self.data['energy'][i] = atoms.get_potential_energy()
            self.data['sigma'][i] = atoms.calc.get_uncertainty()

        unc = np.abs(self.data['sigma']).max()

        if unc > 0.2:
            raise Converged
