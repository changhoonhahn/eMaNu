'''

script to generate class ini files 


'''
import os 
from emanu import util as UT 

def script_Pm_theta(theta): 
    ''' generate CLASS parameter ini files for quijote thetas. 
    **Does not support changes in Mnu**. Mnu is done separately. 
    '''
    # fiducial parameters 
    params = {
            'Omega_cdm': 0.3175 - 0.049,
            'Omega_b': 0.049, 
            'Omega_k': 0.0, 
            'h': 0.6711, 
            'n_s': 0.9624, 
            'k_pivot': 0.05, 
            'sigma8': 0.834,
            'N_eff': 3.046, 
            'P_k_max_1/Mpc':20.0, 
            'z_pk': 0.}
    if theta == 'fiducial': 
        pass 
    elif theta == 'Ob_m': 
        params['Omega_b'] = 0.048
    elif theta == 'Ob_p': 
        params['Omega_b'] = 0.050
    elif theta == 'Om_m': 
        params['Omega_cdm'] = 0.3075 - 0.049
    elif theta == 'Om_p': 
        params['Omega_cdm'] = 0.3275 - 0.049
    elif theta == 'h_m': 
        params['h'] = 0.6911
    elif theta == 'h_p':
        params['h'] = 0.6511
    elif theta == 'ns_m': 
        params['n_s'] = 0.9424
    elif theta == 'ns_p': 
        params['n_s'] = 0.9824
    elif theta == 's8_m': 
        params['sigma8'] = 0.819
    elif theta == 's8_p': 
        params['sigma8'] = 0.849
    else: 
        print('%s not implemented' % theta) 
        raise ValueError
    out_str = '%s' % theta 
    
    job = '\n'.join([
        '# this file achieves maximum precision for the CMB: each ClTT and ClEE, lensed and unlensed, are stable at the 0.01% level',
        'h = %.4f' % params['h'],
        'T_cmb = 2.7255',
        'Omega_b = %.4f' % params['Omega_b'],
        'N_ur = 3.046',
        'Omega_cdm = %.8f' % params['Omega_cdm'],
        'Omega_dcdmdr = 0.0',
        'Gamma_dcdm = 0.0',
        'N_ncdm = 0',
        'Omega_k = 0.',
        'Omega_fld = 0',
        'Omega_scf = 0',
        'YHe = BBN',
        'recombination = RECFAST',
        'reio_parametrization = reio_camb',
        'z_reio = 11.357',
        'reionization_exponent = 1.5',
        'reionization_width = 0.5',
        'helium_fullreio_redshift = 3.5',
        'helium_fullreio_width = 0.5',
        'annihilation = 0.',
        'decay = 0.',
        'output = ,mPk,dTk',
        'modes = s',
        'lensing = no',
        'ic = ad',
        'P_k_ini type = analytic_Pk',
        'k_pivot = 0.05',
        'sigma8 = %.4f' % params['sigma8'],
        'n_s = %.4f' % params['n_s'],
        'alpha_s = 0.',
        'P_k_max_h/Mpc = 10',
        'z_pk = 0',
        'root = output/%s_' % out_str,
        'headers = yes',
        'format = class',
        'write background = y',
        'write thermodynamics = no',
        'write primordial = no',
        'write parameters = yeap',
        'input_verbose = 1',
        'background_verbose = 1',
        'thermodynamics_verbose = 1',
        'perturbations_verbose = 1',
        'transfer_verbose = 1',
        'primordial_verbose = 1',
        'spectra_verbose = 1',
        'nonlinear_verbose = 1',
        'lensing_verbose = 1',
        'output_verbose = 1',
        'k_per_decade_for_pk = 50',
        'k_per_decade_for_bao = 100',
        '',
        'tol_ncdm_bg = 1.e-10',
        '',
        'recfast_Nz0=100000',
        'tol_thermo_integration=1.e-5',
        '',
        'recfast_x_He0_trigger_delta = 0.01',
        'recfast_x_H0_trigger_delta = 0.01',
        '',
        'evolver=0',
        '',
        'k_min_tau0=0.002',
        'k_max_tau0_over_l_max=3.',
        'k_step_sub=0.015',
        'k_step_super=0.0001',
        'k_step_super_reduction=0.1',
        '',
        'start_small_k_at_tau_c_over_tau_h = 0.0004',
        'start_large_k_at_tau_h_over_tau_k = 0.05',
        'tight_coupling_trigger_tau_c_over_tau_h=0.005',
        'tight_coupling_trigger_tau_c_over_tau_k=0.008',
        'start_sources_at_tau_c_over_tau_h = 0.006',
        '',
        'l_max_g=50',
        'l_max_pol_g=25',
        'l_max_ur=150',
        'l_max_ncdm=50',
        '',
        'tol_perturb_integration=1.e-6',
        'perturb_sampling_stepsize=0.01',
        '',
        'radiation_streaming_approximation = 2',
        'radiation_streaming_trigger_tau_over_tau_k = 240.',
        'radiation_streaming_trigger_tau_c_over_tau = 100.',
        '',
        'ur_fluid_approximation = 2',
        'ur_fluid_trigger_tau_over_tau_k = 50.',
        '',
        'ncdm_fluid_approximation = 3',
        'ncdm_fluid_trigger_tau_over_tau_k = 51.',
        '',
        'tol_ncdm_synchronous = 1.e-10',
        'tol_ncdm_newtonian = 1.e-10',
        '',
        'l_logstep=1.026',
        'l_linstep=25',
        '',
        'hyper_sampling_flat = 12.',
        'hyper_sampling_curved_low_nu = 10.',
        'hyper_sampling_curved_high_nu = 10.',
        'hyper_nu_sampling_step = 10.',
        'hyper_phi_min_abs = 1.e-10',
        'hyper_x_tol = 1.e-4',
        'hyper_flat_approximation_nu = 1.e6',
        'q_linstep=0.20',
        'q_logstep_spline= 20.',
        'q_logstep_trapzd = 0.5',
        'q_numstep_transition = 250',
        'transfer_neglect_delta_k_S_t0 = 100.',
        'transfer_neglect_delta_k_S_t1 = 100.',
        'transfer_neglect_delta_k_S_t2 = 100.',
        'transfer_neglect_delta_k_S_e = 100.',
        'transfer_neglect_delta_k_V_t1 = 100.',
        'transfer_neglect_delta_k_V_t2 = 100.',
        'transfer_neglect_delta_k_V_e = 100.',
        'transfer_neglect_delta_k_V_b = 100.',
        'transfer_neglect_delta_k_T_t2 = 100.',
        'transfer_neglect_delta_k_T_e = 100.',
        'transfer_neglect_delta_k_T_b = 100.',
        'neglect_CMB_sources_below_visibility = 1.e-30',
        'transfer_neglect_late_source = 3000.',
        'halofit_k_per_decade = 3000.',
        'l_switch_limber = 40.',
        'accurate_lensing=1',
        'num_mu_minus_lmax = 1000.',
        'delta_l_max = 1000.']) 

    fjob = os.path.join(UT.dat_dir(), 'lt', '%s.ini' % out_str) 
    jb = open(fjob, 'w') 
    jb.write(job) 
    jb.close() 
    return None 


if __name__=="__main__": 
    for tt in ['Om', 'Ob', 'h', 'ns', 's8']: 
        script_Pm_theta('%s_m' % tt)
        script_Pm_theta('%s_p' % tt)
