import time

import cantera as ct
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('./data/latex.mplstyle')

import reactor.helpers as h
import reactor.constants as c

class PlugFlowReactor:
    """
    Class for plug flow reactor methane-air flame implemented with the cantera 
    library.
    \Phi CH4 + 2(O2 + 3.76N2) -> CO2 + 2H2O + 7.52N2
    """
    def __init__(self, T_0=c.T_0, p_0=c.P_0, phi=c.PHI, length=c.LENGTH):
        """
        Initialization of plug flow reactor as a function of initial temperature,
        pressure and equivalence ratio.

        Parameters
        ----------
        T_0 : float
            initial temperature, default as 1200 K.
        p_0 : float
            initial pressure, default as 101325 Pa (1 atm).
        phi : float
            equivalence ratio, default as 1 for stoichiometric composition.
        length : float
            length for the flow reactor, which should be determined according to
            the desired residence time. Default as 6e-4m for 100ms of residence
            time.
        """
        self.T_0 = T_0
        self.p_0 = p_0
        self.phi = phi
        self.length = length 
        self.u_0 = c.U_0
        self.area = c.AREA

        # Set composition string for cantera solution, considering standard
        #oxidizer of O2:3.76N2
        self.comp = h.set_composition(self.phi)
        
    def change_oxidizer(self, comp):
        """
        For brevity, composition calculation will be performed by hand for the
        three following oxidizer options:
            1 - pure O2
            2 - X_O2=0.207, X_N2=0.79, X_OH=0.003
            3 - X_O2=0.207, X_N2=0.79, X_H2O=0.003
        Final composition to be inputed:
            1 - \Phi CH4 + 2O2 -> CO2 + 2H2O
            2 - \Phi CH4 + 1.993(O2 + 3.816N2 + 0.015OH) -> CO2 + 2.016H2O
            3 - \Phi CH4 + 2(O2 + 3.816N2 + 0.015H2O) -> CO2 + 2.03H2O
        """
        self.comp = comp

    def compute_solution(self, mech_file='./data/mech-FFCM1.yaml', n_steps=2000):
        """
        For the specified mechanism, compute the solution using lagrangian
        particle simulation.
        """
        # Import gas mechanism and set initial conditions in cantera:
        gas = ct.Solution(mech_file)
        gas.TPX = self.T_0, self.p_0, self.comp
        mass_flow_rate = self.u_0 * gas.density * self.area

        # Create reactor
        reactor = ct.IdealGasConstPressureReactor(gas)
        # Reactor network for time integration
        simulation = ct.ReactorNet([reactor])

        # Time step resolution
        t_total = self.length / self.u_0
        dt = t_total / n_steps

        # Vectors for space, time and velocity
        ts = (np.arange(n_steps) + 1) * dt
        zs = np.zeros_like(ts)
        us = np.zeros_like(ts)
        states = ct.SolutionArray(reactor.thermo)

        # Loop over time
        for i, t in enumerate(ts):
            # Time integration
            simulation.advance(t)
            # Velocity and respective space
            us[i] = mass_flow_rate / self.area / reactor.thermo.density
            zs[i] = zs[i-1] + us[i] * dt
            states.append(reactor.thermo.state)

        self.results = {
            'space': zs,
            'space_norm': zs/self.length,
            'time': ts,
            'temperature': states.T,
            'X_CH4': states.X[:, gas.species_index('CH4')],
            'X_O2': states.X[:, gas.species_index('O2')],
            'X_OH': states.X[:, gas.species_index('OH')],
            'X_H2': states.X[:, gas.species_index('H2')],
            'X_CO': states.X[:, gas.species_index('CO')],
            'X_H2O': states.X[:, gas.species_index('H2O')],
            'X_CO2': states.X[:, gas.species_index('CO2')],
        }

    def plot_ignition_delay(self):
        """
        Plot  for the plug flow reactor methane-air flame.
            > temperature/molefraction vs space.
        """
        # Temperature vs time
        fig1, ax1 = plt.subplots()
        ax1.plot(self.results['time'], self.results['temperature'])
        ax1.set(
            xlabel = 'Time [s]',
            xscale = 'log',
            ylabel = 'Temperature [K]'
        )
        fig1.savefig(
            f'./results/pfr_t_T_{self.p_0}_{self.T_0}_{self.phi}.svg',
            format='svg',
            bbox_inches='tight'
        )

    def plot_ignition_delay_comp(self):
        """
        Plot  for the plug flow reactor methane-air flame.
            > temperature/molefraction vs space.
        """
        # Composition vs time
        fig2, ax2_a = plt.subplots()
        ax2_b = ax2_a.twinx()
        ax2_a.plot(self.results['space_norm'], self.results['X_CH4'], label=r'X$_{\text{CH}_4}$')
        ax2_a.plot(self.results['space_norm'], self.results['X_O2'], label=r'X$_{\text{O}_2}$')
        ax2_a.plot(self.results['space_norm'], self.results['X_OH'], label=r'X$_{\text{OH}}$')
        ax2_a.plot(self.results['space_norm'], self.results['X_H2'], label=r'X$_{\text{H}_2}$')
        ax2_a.plot(self.results['space_norm'], self.results['X_CO'], label=r'X$_{\text{CO}}$')
        ax2_a.plot(self.results['space_norm'], self.results['X_H2O'], label=r'X$_{\text{H}_2\text{O}}$')
        ax2_a.plot(self.results['space_norm'], self.results['X_CO2'], label=r'X$_{\text{CO}_2}$')
        ax2_a.set(
            xlabel = 'Normalized distance [-]',
            #xscale = 'log',
            xlim = (0.0, 1.0),
            ylabel = 'Mole fraction [-]',
            #yscale = 'log',
        )
        ax2_a.legend(loc=7)
        ax2_b.plot(self.results['space_norm'], self.results['temperature'], '--', color='k')
        ax2_b.set(
            ylabel = 'Temperature [K]'
        )
        fig2.savefig(
            f'./results/pfr_s_X_{int(self.p_0/ct.one_atm)}_{self.T_0}_{self.phi}.svg',
            format='svg',
            bbox_inches='tight'
        )

class WellStirredReactor:
    """
    Class for well stirred reactor of a methane-air flame implemented with the cantera 
    library.
    \Phi CH4 + 2(O2 + 3.76N2) -> CO2 + 2H2O + 7.52N2
    """
    def __init__(self, T_0=c.T_0, p_0=c.P_0, phi=c.PHI, res_time=c.RESIDENCE_TIME):
        """
        Initialization of plug flow reactor as a function of initial temperature,
        pressure and equivalence ratio.

        Parameters
        ----------
        T_0 = float
            initial temperature, default as 1200 K.
        p_0 = float
            initial pressure, default as 101325 Pa (1 atm).
        phi = float
            equivalence ratio, default as 1 for stoichiometric composition.
        """
        self.T_0 = T_0
        self.p_0 = p_0
        self.phi = phi
        self.res_time = res_time
        self.volume = c.VOLUME

        # Set composition string for cantera solution
        self.comp = h.set_composition(self.phi)

    def compute_solution(self, mech_file='./data/gri30.yaml'):
        """
        For the specified mechanism, compute the solution. Consider the following
        arrangement for the well stirred reactor:

            Mixture tank > Mass flow controller > stirred reactor
                > pressure valve > capture tank
        """
        # Import gas mechanism and set inlet conditions in cantera:
        gas = ct.Solution(mech_file)
        gas.TPX = self.T_0, self.p_0, self.comp

        # Create the whole arrangement for the reactor
        inlet = ct.Reservoir(gas)
        gas.equilibrate('HP')
        reactor = ct.IdealGasReactor(gas, volume=self.volume)
        exhaust = ct.Reservoir(gas)
        inlet_mfc = ct.MassFlowController(
            upstream=inlet,
            downstream=reactor,
            mdot=h.mdot, # not working properly to vary the mass flow depending on the residence time
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
        #t_total = 100 * self.res_time # [s]

        # Loop over residence times
        #tic = time.time()
        #t = 0
        counter = 1
        residence_time = self.res_time
        while reactor.T > 500:
        #while (reactor.T > 500) and (counter < 50):
            simulation.set_initial_time(0.0)
            simulation.advance_to_steady_state()
            print(f'tres: {residence_time:.2e}; T: {reactor.T}')
            states.append(reactor.thermo.state, tres=residence_time)
            residence_time *= 0.9
            counter += 1

        self.results = {
            'res_time': states.tres,
            'temperature': states.T,
            'X_CO': states.X[:, gas.species_index('CO')],
            'X_CO2': states.X[:, gas.species_index('CO2')],
            'X_NO': states.X[:, gas.species_index('NO')],
        }

        print(
            self.results['res_time'], 
            self.results['X_CO'], self.results['X_CO2'], self.results['X_NO']
        )

    def plot_ignition_delay(self):
        """
        Plot  for the plug flow reactor methane-air flame.
            > temperature vs time;
            > temperature/molefraction vs space.
        """
        # Temperature vs time
        fig1, ax1 = plt.subplots()
        ax1.plot(self.results['time'], self.results['temperature'])
        ax1.set(
            xlabel = 'Time [s]',
            xscale = 'log',
            ylabel = 'Temperature [K]'
        )

         
        fig2, ax2_a = plt.subplots()
        ax2_b = ax2_a.twinx()
        ax2_a.plot(self.results['time'], self.results['X_CH4'], label=r'X$_{\text{CH}_4}$')
        ax2_a.plot(self.results['time'], self.results['X_O2'], label=r'X$_{\text{O}_2}$')
        ax2_a.plot(self.results['time'], self.results['X_OH'], label=r'X$_{\text{OH}}$')
        ax2_a.plot(self.results['time'], self.results['X_H2'], label=r'X$_{\text{H}_2}$')
        ax2_a.plot(self.results['time'], self.results['X_CO'], label=r'X$_{\text{CO}}$')
        ax2_a.plot(self.results['time'], self.results['X_H2O'], label=r'X$_{\text{H}_2\text{O}}$')
        ax2_a.plot(self.results['time'], self.results['X_CO2'], label=r'X$_{\text{CO}_2}$')
        ax2_a.set(
            xlabel = 'Time [s]',
            #xscale = 'log',
            xlim = (0.00007, 0.0002),
            ylabel = 'Mole fraction [-]'
        )
        ax2_a.legend()
        ax2_b.plot(self.results['time'], self.results['temperature'])
        ax2_b.set(
            ylabel = 'Temperature [K]'
        )

        plt.show()
