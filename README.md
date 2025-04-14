# Orbits Simulator: Planetary Motion Around a Black Hole

A Python package to simulate, visualize, and animate the orbit of a planet around a black hole using various numerical ODE solvers. Includes options for classical and relativistic gravitational models, as well as static and animated visualizations of the orbit.

## Features

- Classical and relativistic orbit simulation modes
- Customizable numerical integration methods: `Trapezoidal`, `RK3`, and `DOP853`
- Configurable system parameters (mass, eccentricity, etc.)
- Save and load solutions
- Initial condition map generation
- Orbit animation and plotting

## Installation

Run the following comand inside the module directory.
```bash
pip install -e .
```
## Usage

### 1. Comand Line Usage

```bash
python orbits.py -e 0.0 -M 5.e-6 -a 1.0 -N 2 -n 250 -s Classical -i RK3 -f sim02 -im -gif
```
This will define the parameters as follow:

- Eccentricity: 0.0

- Black Hole Mass: 5.e-6

- Major-axis: 1.0

- Number of orbital periods: 2 

- Time steps: 250

- Simulation type: Classical 

- Integrator Method: RK3

- File name: sim02

- Save the intial map: True

- Save a gif: True

### 2. Using the INI file (config_orbits.ini)

```bash
python orbits.py --config path/config_orbits.ini
```

## Author 

Developed by Alan Palma as part of the Midterm Exam of Computational Physics II at Yachay Tech University to simulate orbital dynamics in strong gravitational fields.

Email: alan.palma@yachaytech.edu.ec

## License

Licensed under the MIT License.

