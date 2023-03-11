from reactor.reactor import PlugFlowReactor, WellStirredReactor

if __name__ == "__main__":
    #pfr = PlugFlowReactor(T_0=1800, phi=1.0)
    #pfr.compute_solution(mech_file='./data/mech-FFCM1.yaml', n_steps=5000)
    #pfr.plot_ignition_delay() 

    wst = WellStirredReactor(T_0=300, phi=0.85)
    wst.compute_solution(mech_file='./data/gri30.yaml')
