from catlearn.regression import GaussianProcess
import numpy as np
import ase.neb
import ase


class NEB(object):

    def __init__(
            self,
            trajectory,
            nimages=5,
            calculator=None):
        """Bayesian optimization for nudged elastic band.

        Parameters
        ----------
        trajectory: list of Atoms objects
            Initial end-point of the NEB path or Atoms object.
        nimages: int | float
            Number of images of the path, including the end-points.
        calculator: Calculator object
            ASE calculator as implemented in ASE.
            See https://wiki.fysik.dtu.dk/ase/ase/calculators/calculators.html
        """
        self.calculator = calculator
        self.nimages = nimages
        self.iteration = 0
        self.gp = None

        # Produce an interpolations
        self.images = [trajectory[0]]
        for i in range(self.nimages - 2):
            self.images += [trajectory[0].copy()]
        self.images += [trajectory[-1]]

        neb = ase.neb.NEB(self.images)
        neb.interpolate(method='idpp')

        is_pos = self.images[0].get_positions().flatten()
        fs_pos = self.images[-1].get_positions().flatten()
        self.path_distance = np.abs(np.linalg.norm(is_pos - fs_pos))

        cons = self.images[0].constraints[0].todict()['kwargs']['indices']
        self.mask = ~np.in1d(range(len(trajectory[0])), cons)

        # Prepare the training data
        self.train = {'positions': [], 'forces': [], 'energy': []}
        for i, atoms in enumerate(trajectory):
            self.update_train_data(atoms, append=False)

    def run(self, fmax=0.05, umax=0.05, steps=200, gpsteps=750):
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
        if len(self.train['energy']) == 2:
            target = self.nimages // 2
            atoms = self.get_calculator_energy(self.images[target])
            self.update_train_data(atoms)

        previous_convergence = False
        for i in range(steps):

            # 1. Train surrogate model
            self.train_gp_model()

            max_target = max(self.train['energy'])
            for atoms in self.images[1:-1]:
                calc = gp.Calculator(gp=self.gp, scaling=max_target)
                atoms.set_calculator(calc)

            # 2. Perform NEB on surrogate model
            neb = ase.neb.NEB(
                self.images,
                climb=True,
                method='improvedtangent')
            opt = ase.optimize.MDMin(neb, logfile=None)
            opt.run(fmax=fmax, steps=steps)
            print('GP Evalulatons: {}'.format(opt.__dict__.get('nsteps')))

            path = self.get_path_predictions()

            # Check convergence
            max_force = np.abs(self.train['forces'][-1].max())
            force_converged = max_force < fmax
            uncertainty_converged = path['max_simga'] < umax
            if force_converged:
                previous_convergence = True

                if uncertainty_converged:
                    ase.io.write('neb.json', self.images)
                    break

            # Acquisition function
            if not uncertainty_converged or previous_convergence:
                target = np.argmax(path['uncertainty']) + 1
            else:
                target = np.argmax(path['uncertainty'] + path['energy']) + 1

            # Evaluate a new target
            atoms = self.get_calculator_energy(self.images[target])
            self.update_train_data(atoms)

    def get_calculator_energy(self, atoms):
        """Evaluates the energy and forces of the point of interest
        for a given atomistic structure using the provided calculator.

        Parameters
        ----------
        atoms : Atoms object
            Image to be evaluated using provided calculator.

        Returns
        -------
        atoms : Atoms object
            Atoms object with completed calculation.
        """
        atoms = atoms.copy()
        atoms.set_calculator(self.calculator)
        atoms.get_potential_energy()

        self.iteration += 1
        print('Evaluation {}'.format(self.iteration))

        return atoms

    def update_train_data(self, atoms, append=True):
        """Update the training data for the Gaussian process.

        Parameters
        ----------
        atoms : Atoms object
            Configuration to include in the training data.
        append : bool | None
            Whether to write the results to disk and how to append.
        """
        positions = atoms.get_positions()[self.mask].flatten()
        self.train['positions'] += [positions]
        energy = atoms.get_potential_energy()
        self.train['energy'] += [energy]
        forces = -atoms.get_forces()[self.mask].flatten()
        self.train['forces'] += [forces]

        if append is not None:
            ase.io.write('evaluated.json', atoms, append=append)
            ase.io.write('paths.json', self.images, append=append)

    def train_gp_model(self):
        """Train a surrogate Gaussian Process model"""
        train = np.array(self.train['positions'])
        gradients = np.array(self.train['forces'])
        targets = np.array(self.train['energy'])
        scaled_targets = targets - targets.max()
        sigma = np.max([np.std(scaled_targets)**2, 1e-3])

        kdict = [{
            'type': 'gaussian',
            'width': self.path_distance / 2,
            'dimension': 'single',
            'bounds': ((0.1, self.path_distance),),
            'scaling': sigma,
            'scaling_bounds': ((1e-3, 1e2),)
        }]

        self.gp = GaussianProcess(
            kernel_list=kdict,
            regularization=0.0,
            regularization_bounds=(0.0, 0.0),
            train_fp=train,
            train_target=scaled_targets,
            gradients=gradients,
            optimize_hyperparameters=False,
            scale_data=False
        )
        self.gp.optimize_hyperparameters(global_opt=False)

    def get_path_predictions(self):
        """Obtain results from the predicted NEB.

        Returns
        -------
        path : dict
            Energy and uncertainty predictions for NEB images
        """
        path = {
            'energy': np.empty(self.nimages - 2),
            'uncertainty': np.empty(self.nimages - 2)
        }
        for i, atoms in enumerate(self.images[1:-1]):
            pos = [atoms.get_positions()[self.mask].flatten()]
            u = self.gp.predict(test_fp=pos, uncertainty=True)
            path['energy'][i] = atoms.get_total_energy()
            path['uncertainty'][i] = 2 * u['uncertainty_with_reg'][0]
        path['max_simga'] = path['uncertainty'].max()

        return path
