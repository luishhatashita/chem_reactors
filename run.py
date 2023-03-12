import cantera as ct
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('./data/latex.mplstyle')

from reactor.reactor import PlugFlowReactor, WellStirredReactor

def plot_inf_temp(p, Ts, phi, results):
    fig, ax = plt.subplots()
    for T in Ts:
        ax.plot(results[f'{T}']['time'], results[f'{T}']['temp'], label=f'{T} K')
    ax.set(
        xlabel = 'Time [s]',
        xscale = 'log',
        ylabel = 'Temperature [K]'
    )
    ax.legend()
    fig.savefig(
        f'./results/pfr_t_T_{int(p/ct.one_atm)}_all_{int(phi)}.svg',
        format='svg',
        bbox_inches='tight'
    )
    
def plot_inf_p(ps, T, phi, results):
    fig, ax = plt.subplots()
    for p in ps:
        p_atm = int(p/ct.one_atm)
        ax.plot(results[f'{p}']['time'], results[f'{p}']['temp'], label=f'{p_atm} atm')
    ax.set(
        xlabel = 'Time [s]',
        xscale = 'log',
        ylabel = 'Temperature [K]'
    )
    ax.legend()
    fig.savefig(
        f'./results/pfr_t_T_all_{T}_{int(phi)}.svg',
        format='svg',
        bbox_inches='tight'
    )

def plot_inf_eq(p, T, phis, results):
    fig, ax = plt.subplots()
    for phi in phis:
        ax.plot(results[f'{phi}']['time'], results[f'{phi}']['temp'], label=f'$\Phi = {phi}$')
    ax.set(
        xlabel = 'Time [s]',
        xscale = 'log',
        ylabel = 'Temperature [K]'
    )
    ax.legend()
    fig.savefig(
        f'./results/pfr_t_T_{int(p/ct.one_atm)}_{T}_all.svg',
        format='svg',
        bbox_inches='tight'
    )

def plot_inf_oxi(p, T, phi, comps, results):
    fig, ax = plt.subplots()
    for i, comp in enumerate(comps):
        ax.plot(results[f'{i}']['time'], results[f'{i}']['temp'], label=f'Oxidizer ({i+1})')
    ax.set(
        xlabel = 'Time [s]',
        xscale = 'log',
        ylabel = 'Temperature [K]'
    )
    ax.legend()
    fig.savefig(
        f'./results/pfr_t_T_{int(p/ct.one_atm)}_{T}_{phi}_oxis.svg',
        format='svg',
        bbox_inches='tight'
    )

if __name__ == "__main__":
    # 1 - Plug flow reactor
    # a) Autoignition delay time as function of initial pressure and temperature
    ps = np.array([1, 10])*ct.one_atm
    Ts = np.array([1200, 1400, 1600, 1800])
    phis = np.array([0.5, 1, 1.5])

    # i. influence of temperature
    # 1atm at stoichiometric composition
    results_1_all_1 = {}
    for T in Ts:
        pfr = PlugFlowReactor(p_0=ps[0], T_0=T, phi=phis[1])
        pfr.compute_solution(mech_file='./data/mech-FFCM1.yaml', n_steps=5000)
        results_1_all_1[f'{T}'] = {
            'time':pfr.results['time'],
            'temp':pfr.results['temperature']
        }

    plot_inf_temp(ps[0], Ts, phis[1], results_1_all_1)

    # 10 atm at stoichiometric composition
    results_10_all_1 = {}
    for T in Ts:
        pfr = PlugFlowReactor(p_0=ps[1], T_0=T, phi=phis[1], length=6e-5)
        pfr.compute_solution(mech_file='./data/mech-FFCM1.yaml', n_steps=5000)
        results_10_all_1[f'{T}'] = {
            'time':pfr.results['time'],
            'temp':pfr.results['temperature']
        }

    plot_inf_temp(ps[1], Ts, phis[1], results_10_all_1)

    print("--Influence of temperature figures OK!")

    # ii. influence of pressure
    # 1200K at stoichiometric
    results_all_1200_1 = {}
    for p in ps:
        pfr = PlugFlowReactor(p_0=p, T_0=Ts[0], phi=phis[1])
        pfr.compute_solution(mech_file='./data/mech-FFCM1.yaml', n_steps=5000)
        results_all_1200_1[f'{p}'] = {
            'time':pfr.results['time'],
            'temp':pfr.results['temperature']
        }

    plot_inf_p(ps, Ts[0], phis[1], results_all_1200_1)

    # 1800K at stoichiometric
    results_all_1800_1 = {}
    for p in ps:
        pfr = PlugFlowReactor(p_0=p, T_0=Ts[-1], phi=phis[1], length=6e-5)
        pfr.compute_solution(mech_file='./data/mech-FFCM1.yaml', n_steps=5000)
        results_all_1800_1[f'{p}'] = {
            'time':pfr.results['time'],
            'temp':pfr.results['temperature']
        }

    plot_inf_p(ps, Ts[-1], phis[1], results_all_1800_1)

    print("--Influence of pressure figures OK!")
    # ii. influence of equivalence ratio
    # 1200K and 1 atm
    results_1_1200_all = {}
    for phi in phis:
        pfr = PlugFlowReactor(p_0=ps[0], T_0=Ts[0], phi=phi)
        pfr.compute_solution(mech_file='./data/mech-FFCM1.yaml', n_steps=5000)
        results_1_1200_all[f'{phi}'] = {
            'time':pfr.results['time'],
            'temp':pfr.results['temperature']
        }

    plot_inf_eq(ps[0], Ts[0], phis, results_1_1200_all)

    # 1200K and 10 atm
    results_10_1200_all = {}
    for phi in phis:
        pfr = PlugFlowReactor(p_0=ps[-1], T_0=Ts[0], phi=phi)
        pfr.compute_solution(mech_file='./data/mech-FFCM1.yaml', n_steps=5000)
        results_10_1200_all[f'{phi}'] = {
            'time':pfr.results['time'],
            'temp':pfr.results['temperature']
        }

    plot_inf_eq(ps[-1], Ts[0], phis, results_10_1200_all)

    # 1800K and 1 atm
    results_1_1800_all = {}
    for phi in phis:
        pfr = PlugFlowReactor(p_0=ps[0], T_0=Ts[-1], phi=phi, length=6e-5)
        pfr.compute_solution(mech_file='./data/mech-FFCM1.yaml', n_steps=5000)
        results_1_1800_all[f'{phi}'] = {
            'time':pfr.results['time'],
            'temp':pfr.results['temperature']
        }

    plot_inf_eq(ps[0], Ts[-1], phis, results_1_1800_all)

    # 1200K and 10 atm
    results_10_1800_all = {}
    for phi in phis:
        pfr = PlugFlowReactor(p_0=ps[-1], T_0=Ts[-1], phi=phi, length=6e-5)
        pfr.compute_solution(mech_file='./data/mech-FFCM1.yaml', n_steps=5000)
        results_10_1800_all[f'{phi}'] = {
            'time':pfr.results['time'],
            'temp':pfr.results['temperature']
        }

    plot_inf_eq(ps[-1], Ts[-1], phis, results_10_1800_all)

    print("--Influence of equivalence ratio figures OK!")

    # iv. species concentrations at 1 atm, 1400K and \Phi=1.5
    pfr = PlugFlowReactor(p_0=ps[0], T_0=Ts[1], phi=phis[-1])
    pfr.compute_solution(mech_file='./data/mech-FFCM1.yaml', n_steps=5000)
    pfr.plot_ignition_delay_comp()

    print("--Composition figure OK!")

    # b) Autoignition delay time sensitivity to oxidizer
    # At 1 atm, 1400K, \Phi = 0.5
    # Consider the following oxidizer options:
    #     1 - pure O2
    #     2 - X_O2=0.207, X_N2=0.79, X_OH=0.003
    #     3 - X_O2=0.207, X_N2=0.79, X_H2O=0.003
    # Final composition to be inputed:
    #     1 - \Phi CH4 + 2O2 -> CO2 + 2H2O
    #     2 - \Phi CH4 + 1.993(O2 + 3.816N2 + 0.015OH) -> CO2 + 2.016H2O
    #     3 - \Phi CH4 + 2(O2 + 3.816N2 + 0.015H2O) -> CO2 + 2.03H2O
    comps = [
        'CH4:0.5, O2:2',
        'CH4:0.5, O2:1.993, N2:7.605, OH:0.03',
        'CH4:0.5, O2:2, N2:7.632, OH:0.030',
    ]

    results_1_1400_0_5 = {}
    for i, comp in enumerate(comps):
        pfr = PlugFlowReactor(p_0=ps[0], T_0=Ts[1], phi=phis[0])
        pfr.change_oxidizer(comp)
        pfr.compute_solution(mech_file='./data/mech-FFCM1.yaml', n_steps=5000)
        results_1_1400_0_5[f'{i}'] = {
            'time':pfr.results['time'],
            'temp':pfr.results['temperature']
        }

    plot_inf_oxi(ps[0], Ts[1], phis[0], comps, results_1_1400_0_5)

    print("--Oxidizer sensitivity figure OK!")

    # WellStirredReactor not working properly, since the massflow function is
    #not working properly.
    #wst = WellStirredReactor(T_0=300, phi=0.85)
    #wst.compute_solution(mech_file='./data/gri30.yaml')
