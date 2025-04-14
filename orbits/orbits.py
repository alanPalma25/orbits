# Third party lybraries
import numpy as np 
import matplotlib.pyplot as plt 
import scipy as sp
from scipy.integrate import solve_ivp
import pandas as pd
import scienceplots
import os
import glob
from PIL import Image 
from matplotlib import colormaps
import matplotlib.colors as colors
import argparse
import pytest
import configparser
import multiprocessing as mp

# Define the style
plt.style.use(['science', 'notebook', 'no-latex']) # Use a specific style for figures

class OrbitsIntegrators:
    """
    Class for simulating the motion of a planet  orbiting  a central massive black hole, 
    with support for relativistic corrections and different numerical integrators.
    """

    def __init__(self, a, M, n):
        """
        Initialize the orbital integrator with system parameters.
        Parameters:
            a (float): Semi-major axis or initial radial distance [AU].
            M (float): Mass of the central object (e.g., black hole) in solar masses.
            intial_map (bool): Flag for saving the intial map. 
            n (int): Number of grid points to use in the numerical integration.
        """
        # Constants
        self.G = 39.47842 # AU**3/Mo*yr**2
        self.c = 63197.791 # AU/yr

        if n < 50:
            raise ValueError("The time step number should be at least 50")
        else:
            self.n = n # number of points in the grid
    
    def F_rel(self, t, s):
        """
        Computes the slope of the state vector for a relativistic orbital system, 
        to be used in ODE solvers.
        Inputs:
            t(float): time variable.
            s(array):  state vector of the system.
            d(dic): dictionary containing the parameters of the system.
        Outputs:
            slope(float): array of the computed slope of the system.
        """

        # Adapt the slope function for scipy integrator
        if self.name_method == "DOP853":
            # Reshape the state vector to 2D
            s = s.reshape(2, 2)
            
        # Position and velocity components
        x = s[0, 0]
        y = s[0, 1]
        vx = s[1, 0]
        vy = s[1, 1]

        # Distance from the center of the system
        r = np.sqrt(x**2 + y**2)

        # Magitude of angular momentum L
        L = x*vy - y*vx

        # Slope matrix based on the system of ODEs and paremeters
        m = np.array([[0, 1], 
                      [((-self.G*self.M)/(r**3))*(1+((3*L**2)/((r*self.c)**2))), 0]])

        # Compute the slope vector
        slope = np.dot(m, s)

        # Return the slope vector flattened to 1D for using with scipy integrator
        if self.name_method == "DOP853":
            return slope.flatten()

        return slope
    
    def F_classical(self, t, s):
        """
        Computes the slope of the state vector for a classical orbital system, 
        to be used in ODE solvers.
        Inputs:
            t(float): time variable.
            s(array):  state vector of the system.
            d(dic): dictionary containing the parameters of the system.
        Outputs:
            slope(float): array of the computed slope of the system.
        """

        # Adapt the slope function for scipy integrator 

        if self.name_method == "DOP853":
            # Reshape the state vector to 2D
            s = s.reshape(2, 2)

        # Position and velocity components
        x = s[0, 0]
        y = s[0, 1]

        # Distance from the center of the system
        r = np.sqrt(x**2 + y**2)

        # Slope matrix based on the system of ODEs and paremeters
        m = np.array([[0, 1], 
                    [(-self.G*self.M)/((r)**3), 0]])
        
        # Compute the slope vector
        slope = np.dot(m, s)
        
        # Return the slope vector flattened to 1D for using with scipy integrator
        if self.name_method == "DOP853":
            return slope.flatten()

        return slope

    def trapezoidal_E(self, dt, t, sol):
        """
        Solves an ODE using the explicit trapezoidal Euler method.
        This function implements a predictor-corrector scheme based on the trapezoidal rule 
        to approximate the solution of a first-order differential equation.
        Inputs:
            dt (float): Step size.
            t (array): Time points where the solution is computed.
            sol (array): Array to store the solution.
        Output:
            sol (array): Updated array with the numerical solution of the ODE.
        """

        for j in range(0, len(t) - 1):
            sol[j + 1] = sol[j] + dt*self.F(t[j], sol[j]) # Predictor step
            sol[j + 1] = sol[j] + dt*(self.F(t[j], sol[j]) + self.F(t[j + 1], sol[j + 1]))/2 # Corrector step
            
        return sol
    
    def RK3(self, dt, t, sol):
        """
        Solves an ODE using the third-order Runge-Kutta method.
        This function implements the third-order Runge-Kutta method to approximate the solution of a first-order differential equation.
        Inputs:
            dt (float): Step size.
            t (array): Time points where the solution is computed.
            sol (array): Array to store the solution.
        Output:
            sol (array): Updated array with the numerical solution of the ODE.
        """

        for j in range(0, len(t) - 1):
        
            # Compute RK3 intermediate slopes
            k1 = self.F(t[j], sol[j])  # Slope 1 -> k1
            k2 = self.F(t[j] + dt/2, sol[j] + dt*k1/2) # Slope 2 -> k2
            k3 = self.F(t[j] + dt, sol[j] -dt*k1 + 2.*dt*(k2)) # Slope 3 -> k3
            
            # Compute the weighted sum of slopes for the final approximation
            sol[j + 1] = sol[j] + dt*(k1 + 4.*k2 + k3)/6.

        return sol
    
    def DOP853(self, dt, t, sol):
        """
        Solves an ODE using the scipy integrator "DOP853" which implements 
        eight-order Runge-Kutta method to approximate the solution of a first-order differential equation.
        Inputs:
            dt (float): Step size.
            t (array): Time points where the solution is computed.
            sol (array): Array to store the solution.
        Output:
            sln (array): Updated array with the numerical solution of the ODE.
        """

        sln = solve_ivp(self.F, [t[0], t[-1]] , [self.x0, self.y0,self.vx0, self.vy0],\
                method= "DOP853", t_eval = t, rtol = 1e-8, atol = 1e-8) 
        
        sln = sln.y.T # Extract the solution
        sln = sln.reshape(t.shape[0], 2, 2) # Reshape to be consistent with the other methods
        
        return sln
    

class RunOrbits(OrbitsIntegrators):
    """
    Inherits from OrbitsIntegrators to simulate the motion of a body of a planet
    orbiting a central black hole, including options for relativistic 
    corrections and a variety of ODE solvers from OrbitsIntegrators
    """
    
    def __init__(self, M =1.0, e = 0.0, a = 1.0, N = 5.0, n = 150, simulation = "Classical", initial_map = False):
        """
        Initialize the simulation with system parameters and select the force model.
        
        Inputs:
            M (float): Mass of the central black hole in solar masses.
            e (float): Eccentricity of the orbit.
            a (float): Semi-major axis in astronomical units (AU).
            N (float): Number of orbital periods to simulate.
            n (int): Number of time steps.
            simulation (str): Simulation mode ("Classical" or "Relativistic").
            initial_map (bool): Whether to compute a spatial grid.
        """
    
        # Keep attributes from the base class
        super().__init__(a,  M, n)

        # Define the parameters
        if a <= 0.:
            raise ValueError("Semi-major axis should be larger than 0")
        else:
            self.a = a # Semi-major axis

        if e < 0. or e >=1.:
            raise ValueError("Eccentricity should take a value in the range [0., 1.)")
        else:
            self.e = e # Eccentricity

        if M <=0.:
            raise ValueError("Black hole mass can not be cero or negative")  
        else: 
            self.M = M # Mass of the black hole

        if N<=0.:
            raise ValueError("The number of orbit periods should be larger than 0")
        else:
            self.N = N # Orbital Periods

        # Define the initial conditions 
        self.x0 = 0
        self.y0 = a*(1-e)
        self.vx0 = -np.sqrt((self.G*M*(1+e))/(a*(1-e)))
        self.vy0 = 0

        # Define the simulation type
        self.select_simulation(simulation)
        self.simulation_type = simulation

        # Compute the black hole radious
        self.rs = (2*self.G*self.M)/(self.c**2) 

        # Generate the intial map if desired
        if initial_map:
            self.initial_map()

    def select_method(self, method_name):
        """
        Selects the numerical integrator, ODE solver method, by name.
        Input:
            method_name (str): "Trapezoidal", "RK3", or "DOP853".
        """
        
        # Methods available
        methods = {"Trapezoidal" : self.trapezoidal_E,
                   "RK3" : self.RK3,
                   "DOP853" : self.DOP853}

        # Assign the methood in base of the entry
        if method_name in methods:
            self.method = methods[method_name]
        elif method_name == "":
            raise ValueError("Specify a method")
        else:
            raise ValueError(f" {method_name} is not available") 

    def select_simulation(self, simulation_name):
        """
        Selects the slope function to use in the numerical simulation. 
        Inputs:
            simulation_name (str): "Classical" or "Relativistic".
        """
        
        # Simulation available
        simulations = {"Classical" : self.F_classical,
                       "Relativistic" : self.F_rel}

        # Assign the simulation in base of the entry
        if simulation_name in simulations:
            self.F = simulations[simulation_name]
        elif simulation_name == "":
            raise ValueError("Specify a simulation")
        else:
            raise ValueError(f" {simulation_name} is not available")

    
    def solve_ODE(self, method_name ):
        """
        Solves a system of first-order ODEs using a specified numerical method specified in method_name.
        Inputs:
            method_name (str): The name of the numerical method to use.
        Outputs:
            time (array): Array of time values.
            sol (array): Solution array containing the computed states at each time step.
                         Solution array with shape (n, 2, 2). 
                         x -> sol[:, 0, 0]
                         y -> sol[:, 0, 1]
                         vx -> sol[:, 1, 0]
                         vy -> sol[:, 1, 1]

        """
        # Select the method
        self.name_method = method_name
        self.select_method(method_name)

        # Define the evaluation time
        T = np.sqrt((4.0*np.pi**2*self.a**3)/(self.G*self.M)) # Orbital period
        time = np.linspace(0, self.N*T, self.n) # Time vector
        dT = time[1] - time[0] # Time step


        #Initializate the solution vector
        s_i1 = np.zeros((len(time), 2, 2)) 

        # Add intial conditions
        s0 = np.array([[self.x0, self.y0], [self.vx0, self.vy0]])
        s_i1[0, :, :] = s0 # Assign initial conditions

        # Compute the solution using the trapezoidal method
        sol = self.method(dT, time, s_i1)

        # Define the solution for all class
        self.solution = sol
        self.time_arr = time

        return time, sol
    
    def save_solution(self, filename = "simulation"):
        """
        Save the solution array to a tab-separated output file in 'outputfolder/'.
        Handles overwriting with user confirmation. 
        Inputs:
            filename (str): Name prefix for the output file.
        Returns:
            filename (str): Final filename used to save the solution.
        """

        # Define the name of the directory
        name_dir = "outputfolder" 

        # Check if the directory exists, if not create it
        if os.path.isdir(name_dir):
            print(f"Directory '{name_dir}' already exists.")
        else:
            print(f"Directory '{name_dir}' has been created.")
            os.mkdir(name_dir)

        # Creates a pandas data frame
        solution = pd.DataFrame({"Time[day]": self.time_arr, 
                    "x[AU]": self.solution[:, 0, 0],
                    "y[AU]": self.solution[:, 0, 1],
                    "vx[AU/day]": self.solution[:, 1, 0],
                    "vy[AU/day]": self.solution[:, 1, 1]})

        # Define name acoording the simulation
        filename = filename + "-" + self.simulation_type[:3] 

        # Check if the file already exist
        while os.path.exists(name_dir + "/" + filename + ".out"):

            print(f"{filename}.out already exists")

            # Choose if rewrite or not
            val = str(input(f"Do you want to rewrite {filename}.out? (Yes/No)"))

            # If statement to rewrite or choose a new name
            if val == "yes" or val == "Yes":
                break

            elif val == "No" or val == "no":
                filename = str(input("Save as: "))
                filename = filename + "-" + self.simulation_type[:3] 

            else: 
                print("Invalid option. Please enter 'Yes' or 'No'.")

        # Save the solution to a file
        solution.to_csv(name_dir + "/" + filename + ".out", index = False, sep = "\t", float_format ='{:.6e}'.format)

        # Define file name for all the class 
        self.filename = filename

        print(f"Solution have been saved in '{name_dir}' as: {filename}.out")

        return filename
    
    def initial_map(self):
        """
        Function that allows to the user save an intial visualization of the system.
        Outputs: 
             Shows a circle representing the black hole and planet position at the intial time.
        """
        # Define the name of the directory to save the map
        name_dir = "outputfolder" 

        # Check if the directory exists, if not create it
        if os.path.isdir(name_dir):
            print(f"Directory '{name_dir}' already exists.")
        else:
            print(f"Directory '{name_dir}' has been created.")
            os.mkdir(name_dir)


        # Plot the solution
        fig, ax = plt.subplots(figsize = (10, 8))

        # Define a scaling factor for vector speed representation
        s_fac = 1.5e-5
        # Compute the modulus of velocity
        v = np.sqrt(self.vx0**2 + self.vy0**2)

        # Black hole
        black_hole = plt.Circle((0 ,0), self.rs, facecolor = "black", edgecolor = "darkorange", linewidth = 2.0, label = "Black Hole")
        ax.add_patch(black_hole)
        # Planet
        planet = plt.Circle((self.x0, self.y0), 0.03 , facecolor = "yellowgreen", edgecolor = "olivedrab", linewidth = 1.0, label = "Planet", zorder = 10)
        ax.add_patch(planet)
        # Velocity
         # Velocity vector
        ax.quiver(self.x0, self.y0, s_fac*self.vx0, s_fac*self.vy0, angles = 'xy', scale_units = 'xy',
              scale = 1, color = "red", width = 0.004, label = f"$v$ = {v:.2e} AU/yr")

        ax.set_xlabel("x [AU]")
        ax.set_ylabel("y [AU]")
        ax.set_title(f"Planet and Black Hole System: \n Initial Conditions")

        ax.grid(alpha = 0.2)
        ax.set_xlim(-self.y0 - 0.2, self.y0 + 0.2)
        ax.set_ylim(-self.y0 - 0.2, self.y0 + 0.2)

        ax.set_aspect('equal')  # Ensures circles stay circular
        ax.legend(frameon = True, fontsize = 10, loc = 1)

        plt.savefig(name_dir + "/intial_map-" + self.simulation_type[:3] + ".png")
        print(f"The intial map have been saved in '{name_dir}' as: ", "intial_map-" + self.simulation_type[:3] + ".png")


class AnimateOrbits():
    """
    Class for visualizing and animating the trajectory of a planet orbiting a black hole 
    using precomputed numerical data from an orbit simulation.
    """
        
    def __init__(self, orbit_object):
        """
        Initialize the animation object using an existing orbit simulation instance.

        Inputs:
            orbit_object (OrbitsIntegrators): Instance containing the simulation parameters and metadata.
        """

        # Take data from instance containing the simulation
        self.G = orbit_object.G # AU**3/Mo*yr**2
        self.M = orbit_object.M # M0
        self.c = orbit_object.c # AU/yr
        self.simulation_name = orbit_object.simulation_type # Classical or Relativistic Simulation
        self.name_method = orbit_object.name_method # Define integrator method name
        self.file_name = orbit_object.filename  # Define name of file containing the solution
        self.rs = orbit_object.rs # Define the black hole radious

        # Intialize the method for reading the precomputed data
        self.read_solution()
    
    def read_solution(self):
        """
        Read the orbit simulation results from a file and load them into numpy arrays.
        The file must be located in the 'outputfolder/' directory and formatted with tab separation.
        Expected columns include: Time[day], x[AU], y[AU], vx[AU/day], vy[AU/day].
        """
        # Read the solution from the file
        self.solution = pd.read_csv("outputfolder/" + self.file_name + ".out", sep = "\t", header = 0)
        # Extract time and solution arrays
        self.time_arr = np.array(self.solution["Time[day]"], dtype = float)
        self.x = np.array(self.solution["x[AU]"], dtype = float)
        self.y = np.array(self.solution["y[AU]"], dtype = float)
        self.vx = np.array(self.solution["vx[AU/day]"], dtype = float)
        self.vy = np.array(self.solution["vy[AU/day]"], dtype = float)

        
    def plot_simulation(self):
        """
        Generate a static plot of the planet's orbit around the black hole using the final simulation state.
        
        Outputs:
            Shows the full orbit trajectory and a circle representing the black hole and planet position at the last time step. 
            (Displays a matplotlib figure).
        """

        # Plot the solution
        fig, ax = plt.subplots(figsize = (10, 8))
        

        # Planet orbit
        ax.plot(self.x, self.y, color = "gray", linestyle = "-", linewidth = 0.8, label="Planet Orbit")
        # Black hole
        black_hole = plt.Circle((0 ,0), self.rs, facecolor = "black", edgecolor = "darkorange", linewidth = 2.0, label = "Black Hole")
        ax.add_patch(black_hole)
        # Planet
        planet = plt.Circle((self.x[-1], self.y[-1]), 0.03 , facecolor = "yellowgreen", edgecolor = "olivedrab", linewidth = 1.0, label = "Planet", zorder = 10)
        ax.add_patch(planet)
        # Time stamp
        ax.text(0.04, 0.96, f"Time: {self.time_arr[-1]:.3e} yr", ha ='left', va = 'top', fontsize = 13, 
            bbox = dict(facecolor = 'white', alpha = 0.7, edgecolor = 'black'), transform = ax.transAxes) 

        ax.set_xlabel("x [AU]")
        ax.set_ylabel("y [AU]")
        ax.set_title(f"Planet Orbit Around a Black Hole: \n {self.simulation_name} Simulation Using {self.name_method} Method")

        ax.grid(alpha = 0.2)
        ax.set_xlim(np.min(self.x) - 0.2, np.max(self.x) + 0.2)
        ax.set_ylim(np.min(self.y)-0.2, np.max(self.y)+0.2)

        ax.set_aspect('equal')  # Ensures circles stay circular
        ax.legend(frameon = True, fontsize = 10, loc = 1)

        plt.show()
    
    def plot_for_animate(self, i):
        """
        Generate a single frame of the animation showing the planet's position and velocity at time index i.
        Parameters:
            output_dir (str): Directory where the frame image will be saved.
            i (int): Index of the time step to render.
        Outputs:
            Saves a PNG image of the frame.
        """

        # Index the solution for creating the frame
        x = self.x[:i]
        y = self.y[:i]

        # Define a scaling factor for vector speed representation
        s_fac = 1.5e-5
        # Compute the modulus of velocity
        v = np.sqrt(self.vx[i-1]**2 + self.vy[i-1]**2)
        
        # Creating color map for better speed representation
        max_speed = np.sqrt(np.max(self.vx**2 + self.vy**2)) # Estimate max for normalization
        min_speed = np.sqrt(np.min(self.vx**2 + self.vy**2)) # Estimate min for normalization
        norm = colors.Normalize(vmin = min_speed, vmax = max_speed) # Normalize object (Get values from 0 to 1 for cmap())
        cmap = colormaps.get_cmap("inferno_r") # Choose colormap
        arrow_color = cmap(norm(v)) # Assign a color in cmap to v


        # Plot the solution
        fig, ax = plt.subplots(figsize = (10, 8))

        # Plante Orbit
        ax.plot(x, y, color = "gray", linestyle = "-", linewidth = 0.8, label="Planet Orbit")
        black_hole = plt.Circle((0 ,0), self.rs, facecolor = "black", edgecolor = "darkorange", linewidth = 2.0, label = "Black Hole")
        # Black hole
        ax.add_patch(black_hole)
        # Planet
        planet = plt.Circle((x[-1], y[-1]), 0.03 , facecolor = "yellowgreen", edgecolor = "olivedrab", linewidth = 1.0, label = "Planet", zorder = 10)
        ax.add_patch(planet)
        # Time stamp
        ax.text(0.04, 0.96, f"Time: {self.time_arr[i-1]:.3e} yr", ha = 'left', va = 'top', fontsize = 10, 
            bbox = dict(facecolor = 'white', alpha = 0.7, edgecolor = 'black'), transform = ax.transAxes) 
        # Velocity vector
        ax.quiver(x[-1], y[-1], s_fac*self.vx[i-1], s_fac*self.vy[i-1], angles = 'xy', scale_units = 'xy',
              scale = 1, color = arrow_color, width = 0.004, label = f"$v$ = {v:.2e} AU/yr")


        ax.set_xlabel("x [AU]")
        ax.set_ylabel("y [AU]")
        ax.set_title(f"Planet Orbit Around a Black Hole: \n {self.simulation_name} Simulation Using {self.name_method} Integrator")

        ax.grid(alpha = 0.2)
        ax.set_xlim(np.min(self.x) - 0.2, np.max(self.x) + 0.2)
        ax.set_ylim(np.min(self.y)-0.2, np.max(self.y)+0.2)

        ax.set_aspect('equal')  # Ensures circles stay circular
        ax.legend(frameon = True, fontsize = 10, loc = 1)

        # Save the figure frame
        plt.savefig(self.output_dir + f"/{self.file_name}" + ".{:03d}.png".format(i))   
        plt.close()  
    
    def animate(self):
        """
        Generate an animated GIF from the planet orbit simulation.
            - Creates a folder to store frames.
            - Generates a PNG for each time step.
            - Compiles the PNGs into an animated GIF showing orbital evolution.
        Outputs:
            GIF saved in the output folder.
        """
        # Define the directory name for saving the figures
        name_dir = "outputfolder/" + self.file_name

        # Create a directory for saving the figures and gift 
        if os.path.isdir(name_dir):
            print(f"Directory '{name_dir}/' already exists.")
        else:
            print(f"Directory '{name_dir}/' has been created.")
            os.mkdir(name_dir)

        print("Generating GIF...")
       
        # Create all images

        #start = t.time()

        n_cpu = mp.cpu_count()
        self.output_dir = name_dir # Define output folder

        # Paralelize image generation
        pool = mp.Pool(processes = n_cpu)
        pool.map(self.plot_for_animate, range(1, self.time_arr.shape[0] + 1))

        # for i in range(1, self.time_arr.shape[0] + 1):
        #     self.plot_for_animate(i)
        #end = t.time()
        #print(f'execution time is {end - start} s')

        #Read all the generated figures to create the movie

        #Define the input directory
        images_input = name_dir + f"/{self.file_name}.***.png"

        # Collect the images
        imgs = (Image.open(f) for f in sorted(glob.glob(images_input)))

        img = next(imgs)

        #Define the output directory
        imgif_output = name_dir + f"/{self.file_name}.gif"

        # Save the GIF
        img.save(fp = imgif_output, format="GIF", append_images=imgs,\
                save_all=True, duration = 100, loop = 0)
                    
        return print(f"The movie was generated correctly in '{name_dir}/' as:", f"{self.file_name}.gif")


if __name__ == "__main__" :

    # Parsing the code
    parser = argparse.ArgumentParser(prog = "Orbit Simualtor", 
                                     description = '''\
                                        Simulate a two-body system where a planet is orbiting a black hole. This simulation accounts for relativistic effects, 
                                        the simulation type can be specified ("Classical" or "Relativistic") as well as the integrator method ("trapezoidal_E"  or "RK3" or "DOP853").
                                        An additional option is provided to save a GIF animation that visualizes the evolution of the system over a defined number of orbital periods, N. 
                                        ''',
                                     epilog = "Author: Alan Palma, Date: 06/04/2025")


    
    # Version 
    parser.add_argument("--version", action = "version", version="%(prog)s 1.0.0")

    # Option in case of Init File Provided
    parser.add_argument("-c", "--config", help = "INI file path for running the simulation", type = str, default="config_orbit.ini")




    # Add parameters as arguments
    # Eccentricity
    parser.add_argument("-e", 
                        help = "Eccentricity of the orbit (Value Between 0 and 1)", type = float, default = 0.)
    # Black hole mass
    parser.add_argument("-M", 
                        help = "Mass of the central black hole in solar masses", type = float, default = 5.e-6)
    # Semi-major axis
    parser.add_argument("-a", 
                        help = "emi-major axis of the elliptical relative motion", type = float, default = 1.0)
    # Orbital periods
    parser.add_argument("-N", 
                        help = "Orbital periods for running the simulation", type = int, default = 2)
    # Resolution
    parser.add_argument("-n", help = "Number of time steps.", type = int, default = 250)

    # Define the simulation type
    parser.add_argument("-s", "--stype", choices = ["Classical", "Relativistic"], 
                        help = "Simulation type", type = str, default = "Classical")

    # Define the integrator method
    parser.add_argument("-i", "--integrator", choices = ["Trapezoidal", "RK3", "DOP853"], help = "Integrator method", type = str, default = "DOP853")
    
    # Input the file name
    parser.add_argument("-f", "--filename", help = "Define the output file name", type = str, default = "orbit_simulation") 
    
    # Choise if save intial map
    parser.add_argument("-im", "--init_map", action=argparse.BooleanOptionalAction, default = False, help = "Save a figure of the intial conditions of the system.")    

    # Choise gift creation
    parser.add_argument("-gif", action=argparse.BooleanOptionalAction, default = False, help = "Save GIF of the simulation")


    args = parser.parse_args()

    # If INI file is provided rewrite the arguments
    if args.config and os.path.exists(args.config):
        config = configparser.ConfigParser()
        config.read(args.config)

        # Obligatory parameters
        sim = config["SIMULATION"]
        args.e = float(sim.get("e", args.e))
        args.M = float(sim.get("M", args.M))
        args.a = float(sim.get("a", args.a))
        args.N = int(sim.get("N", args.N))

        # Optional Parameters
        opt = config["OPTIONAL"]
        args.n = int(opt.get("n", args.n))
        args.stype = opt.get("stype", args.stype)
        args.integrator = opt.get("integrator", args.integrator)
        args.filename = opt.get("filename", args.filename)
        args.gif = config.getboolean("OPTIONAL", "gif", fallback=args.gif)
        args.init_map = config.getboolean("OPTIONAL", "initial_map", fallback=args.init_map)

    # Show the arguments 
    print("---------------------------------------Simulation Parameters---------------------------------------")
    print(f"Eccentricity: {args.e}")
    print(f"Black Hole Mass: {args.M}")
    print(f"Semi-major Axis: {args.a}")
    print(f"Periods: {args.N}")
    print(f"Time Steps: {args.n}")
    print(f"Simulation Type: {args.stype}")
    print(f"Integrator: {args.integrator}")
    print(f"Output File: {args.filename}")
    print(f"Save GIF: {args.gif}")
    print(f"Save intial map: {args.init_map}")
    print("---------------------------------------------------------------------------------------------------")

    # Instanciate the class 
    orbit_par = RunOrbits(M = args.M, e = args.e, a = args.a, N = args.N, n = args.n, simulation = args.stype, initial_map = args.init_map)
    #Solve the ODE
    time, sol = orbit_par.solve_ODE(args.integrator)
    # Save the solution
    orbit_par.save_solution(args.filename)

    # Choise if save the GIF
    if args.gif:
        # Instanciate the class for animating
        animate_orbit_par = AnimateOrbits(orbit_par)
        # Run animation method
        animate_orbit_par.animate()
