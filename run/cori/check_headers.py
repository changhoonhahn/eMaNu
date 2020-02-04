import os 
from emanu import util as UT 
from emanu.sims import readgadget as RG

thetas = ['Om_p', 'Ob2_p', 'h_p', 'ns_p', 's8_p', 'Om_m', 'Ob2_m', 'h_m', 'ns_m', 's8_m',
        'Mnu_p', 'Mnu_pp', 'Mnu_ppp', 'fiducial', 'fiducial_ZA']

snapnum = 4 

f = open(os.path.join(UT.code_dir(), 'emanu', 'dat', 'quijote_header_lookup.dat'), 'w')
f.write('# theta snapnum Om Ol z h Hz\n')
for theta in thetas:
    snap_folder = os.path.join(UT.dat_dir(), 'quijote', 'Snapshots', theta, '0')
    # read in Gadget header (~65.1 microsec) 
    header = RG.header(os.path.join(snap_folder, 'snapdir_%s' % str(snapnum).zfill(3), 
        'snap_%s' % str(snapnum).zfill(3)))
    Om  = header.omega_m
    Ol  = header.omega_l
    z   = header.redshift
    h   = header.hubble 
    Hz  = header.Hubble #100.0 * np.sqrt(Om * (1.0 + z)**3 + Ol) # km/s/(Mpc/h)
    f.write('%s\n' % '\t'.join([str(tt) for tt in [theta, snapnum, Om, Ol, z, h, Hz]])) 
f.close() 
