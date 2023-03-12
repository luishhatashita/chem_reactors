import cantera as ct
# Default state variables
T_0 = 1200 # [K]
P_0 = ct.one_atm # [Pa]
PHI = 1.0 # [-]
# Default PFR flow parameters based on sample script from cantera, see pfr.py.
LENGTH = 6e-4 # [m]
U_0 = 0.006 # [m/s]
AREA = 1.0e-4 # [m^2]
# Default WSR flow/geometry parameters based on sample script from cantera, see
#stirred_reactor.ipynb
RESIDENCE_TIME = 0.1 # [s]
VOLUME = 1 # [m^3]
