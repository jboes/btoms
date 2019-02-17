import numpy as np
import ase


class Calculator(ase.calculators.calculator.Calculator):
    """Gaussian Process calculator for ASE"""
    implemented_properties = ['energy', 'forces']

    def __init__(
            self,
            gp,
            scaling,
            finite_step=1e-4,
            **kwargs
    ):
        ase.calculators.calculator.Calculator.__init__(self, **kwargs)

        self.gp = gp
        self.scaling = scaling
        self.fs = finite_step

    def calculate(self, atoms=None, properties=['energy', 'forces'],
                  system_changes=ase.calculators.calculator.all_changes):

        self.atoms = atoms
        if self.atoms.constraints:
            cons = self.atoms.constraints[0].todict()['kwargs']['indices']
            mask = ~np.in1d(range(len(self.atoms)), cons)
        else:
            mask = np.ones(len(self.atoms), dtype=bool)

        ase.calculators.calculator.Calculator.calculate(
            self, atoms, properties, system_changes)

        # Get energy
        test = self.atoms.get_positions()[mask].flatten()[None, :]
        predictions = self.gp.predict(test_fp=test, uncertainty=False)
        energy = predictions['prediction'][0][0] + self.scaling

        # Get forces via finite difference
        diff = np.diag([self.fs] * len(test[0]))
        center = np.tile(test[0], (len(test[0]), 1))

        f_pos = self.gp.predict(test_fp=center + diff)['prediction']
        f_neg = self.gp.predict(test_fp=center - diff)['prediction']
        gradient = (-f_neg + f_pos) / (2 * self.fs)

        forces = np.zeros((len(self.atoms), 3))
        forces[mask] = -gradient.reshape(-1, 3)

        self.results['energy'] = energy
        self.results['forces'] = forces
