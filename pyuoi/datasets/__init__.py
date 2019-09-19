
def load_swimmer(flatten=True):
    from pkg_resources import resource_filename
    from sklearn.preprocessing import minmax_scale
    import h5py
    with h5py.File(resource_filename('pyuoi', 'data/Swimmer.h5'), 'r+') as f:
        Swimmers = f['Y'][:]
    if flatten:
        Swimmers = Swimmers.T.reshape(256, 1024)
    return Swimmers
