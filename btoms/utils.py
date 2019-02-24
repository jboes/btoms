import ase.neb


def get_neb_trajectory(endpoint1, endpoint2=None, nimages=5):
    """Return a NEB trajectory with IDPP interpolation.

    Parameters
    ----------
    endpoint1 : Atoms object | list | str
        The second endpoint for use in the NEB. Can be either an
        atoms object, a path to an atoms object. Alternatively, it
        may be a list or 2 paths or atoms objects if endpoint2 is None.
    endpoint2 : Atoms object | str
        The second endpoint for use in the NEB. Can be either an
        atoms object or a path to an atoms object.
    nimages : int
        Number of images to include in the trajectory.

    Returns
    -------
    images : list of Atoms objects (N,)
        Trajectory of the provided endpoints.
    """
    if endpoint2 is None:
        trajectory = endpoint1
    else:
        trajectory = [endpoint1, endpoint2]

    for i, image in enumerate(trajectory):
        if isinstance(image, str):
            trajectory[i] = ase.io.read(image)

    images = [trajectory[0]]
    for i in range(nimages):
        images += [trajectory[0].copy()]
    images += [trajectory[-1]]

    neb = ase.neb.NEB(images)
    neb.interpolate(method='idpp', mic=True)

    return images
