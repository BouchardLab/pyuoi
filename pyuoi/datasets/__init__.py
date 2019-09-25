
def load_swimmer(flatten=True):
    from pkg_resources import resource_filename
    import h5py
    with h5py.File(resource_filename('pyuoi', 'data/Swimmer.h5'), 'r+') as f:
        swimmers = f['Y'][:].astype(float)
    if flatten:
        swimmers = swimmers.T.reshape(256, 1024)
    return swimmers
