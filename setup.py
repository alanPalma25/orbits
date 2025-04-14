# For installation
from setuptools import setup, find_packages

# Call setup
setup(name = "orbits", 
      description = '''\
        Simulate a two-body system where a planet is orbiting a black hole. This simulation accounts for relativistic effects, 
        the simulation type can be specified ("Classical" or "Relativistic") as well as the integrator method ("trapezoidal_E"  or "RK3" or "DOP853").
        An additional option is provided to save a GIF animation that visualizes the evolution of the system over a defined number of orbital periods, N. 
        ''', 
      author = "Alan I. Palma", license = "MIT",
      version = "1.0.0",
      author_email = "alan.palma@yachaytech.edu.ec", 
      packages = find_packages(), 
      install_requires = ["numpy", "matplotlib", "scipy", "pandas",
                          "scienceplots", "Pillow", "argparse",
                          "pytest", "configparser"])
