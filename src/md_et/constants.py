"""Unit conversion constants between atomic units and ASE units."""

# Length
BOHR_TO_ANG = 0.529177249
ANG_TO_BOHR = 1.0 / BOHR_TO_ANG

# Energy
HARTREE_TO_EV = 27.211386245988

# Forces (energy/length)
HARTREE_BOHR_TO_EV_ANG = HARTREE_TO_EV / BOHR_TO_ANG

# Hessian (energy/length^2)
HARTREE_BOHR2_TO_EV_ANG2 = HARTREE_TO_EV / (BOHR_TO_ANG**2)
