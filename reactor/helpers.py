def set_composition(phi):
    """
    Set composition string for the methane-air flame in the form of: 

        'species_i:n_i'

    Parameters
    ----------
    phi: float
        equivalence ratio of combustion reaction.
    """
    return f'CH4:{phi}, O2:2, N2:7.52'

def mdot(t):
    """
    Variable mass flow rate for well stirred reactor for constant residence time
    in the reactor.
    """
    return reactor.mass / residence_time
