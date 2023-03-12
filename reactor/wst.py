import cantera as ct
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('../data/latex.mplstyle')

gas = ct.Solution('../data/gri30.yaml')

eq_ratio = 0.85
comp = f'CH4:{eq_ratio}, O2:2, N2:7.52'
gas.TPX = 300, ct.one_atm, comp

inlet = ct.Reservoir(gas) 
gas.equilibrate('HP')
reactor = ct.IdealGasReactor(gas, volume=1.0)
exhaust = ct.Reservoir(gas)

def mdot(t):
    return reactor.mass / residence_time

inlet_mfc = ct.MassFlowController(
    upstream=inlet,
    downstream=reactor,
    mdot=mdot,
)
outlet_mfc = ct.PressureController(
    upstream=reactor,
    downstream=exhaust,
    master=inlet_mfc,
    K=0.01,
)
# Reactor network for time integration
simulation = ct.ReactorNet([reactor])

# Time step
states = ct.SolutionArray(gas, extra=['tres']) 

# Loop over residence times
counter = 1
residence_time = 50 
while reactor.T > 500:
#while (reactor.T > 500) and (counter < 50):
    simulation.set_initial_time(0.0)
    simulation.advance_to_steady_state()
    print(f'tres: {residence_time:.2e}; T: {reactor.T}')
    states.append(reactor.thermo.state, tres=residence_time)
    residence_time *= 0.95
    counter += 1

fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
ax1.plot(states.tres, states('CO').X, label=r'CO')
ax1.plot(states.tres, states('CO2').X, label=r'CO$_2$')
ax1.plot(states.tres, states('NO').X, label=r'NO')
ax2.plot(states.tres[:], states.T[:], 'r')
ax1.set(
    xlabel='Residence time [s]',
    xscale='log',
    ylabel='Molar fraction [-]',
)
ax1.legend()
ax2.set_ylabel('Temperature [K]')
fig.savefig(
    '../results/wst_t_X.svg',
    format='svg',
    bbox_inches='tight'
)
