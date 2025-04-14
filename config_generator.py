"""
This python script generates automatically the INI file that can be used for running the simulation of the moduel orbits.
The user should only modify the parametres from the INI file already created. 
"""
# Third party libraries
import configparser

# Create an empty dictionary to store the config_orbit.ini file information
config = configparser.ConfigParser()

# Obligatory parameters
config["SIMULATION"] = {}

config["SIMULATION"]["e"] = "0."
config["SIMULATION"]["M"] = "5.e-6"
config["SIMULATION"]["a"] = "1.0"
config["SIMULATION"]["N"] = "2"

# Optional Parameters
config["OPTIONAL"] = {}

config["OPTIONAL"]["n"] = "250"
config["OPTIONAL"]["stype"] = "Classical"
config["OPTIONAL"]["integrator"] = "DOP853"
config["OPTIONAL"]["filename"] = "orbit_simulation"
config["OPTIONAL"]["gif"] = "False"
config["OPTIONAL"]["initial_map"] = "False"

# Save the config file
with open('config_orbit.ini', 'w') as configfile:
  config.write(configfile)