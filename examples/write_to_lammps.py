""" Example of writing a configuration to a LAMMPS dump file. """
import rumdpy as rp

def main():
    # Generate configuration with a FCC lattice
    conf = rp.make_configuration_fcc(nx=7, ny=7, nz=7, rho=1.0, T=1.2)
    conf.copy_to_device()
    ...  # Do something with the configuration
    conf.copy_to_host()

    # Write configuration to LAMMPS dump file
    lmp_dump = rp.configuration_to_lammps(conf)
    print(lmp_dump, file=open('dump.lammps', 'w'))

if __name__ == '__main__':
    main()
