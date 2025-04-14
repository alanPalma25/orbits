# Third party libraries
import orbit.orbits as orbit
import pytest
import numpy as np

class TestClass():
    """
    A test suite for validating the behavior of m the 'orbit' module.
    Includes tests for input validation, integrator selection, output correctness, and initial conditions.
    """
    @classmethod
    def setup_class(cls):
        """
        Setup method that initializes two orbital simulations with different 
        eccentricities and semi-major axes.
        """    
        # Instanciate the classes for testing
        cls.orbit1 = orbit.RunOrbits(M = 5.e6, e = 0.0, a = 1.0, N = 2.0, n = 250, 
                        simulation = "Relativistic", initial_map = True)
        cls.orbit2 = orbit.RunOrbits(M = 5.e6, e = 0.9, a = 4.0, N = 2.0, n = 250, 
                        simulation = "Relativistic", initial_map = True)

        pass

    @classmethod
    def teardown_class(cls):
        """
        Cleans up initialized orbit instances.
        """
        # Clean after test 
        del cls.orbit1
        del cls.orbit2

        pass

    def test_input_values_module(self):
        """
        Unit test that verifies exceptions are raised for invalid physical parameters.
        Checks edge cases such as negative mass, eccentricity, and semi-major axis,
        as well as an insufficient number of time steps.
        """

        #Define incorrect parameters  
        M = -5.e6
        e = -1.0
        a = -2.0
        N = -2.0
        n = 1

        with pytest.raises(ValueError, match="Black hole mass can not be cero or negative"):
            orbit.RunOrbits(M = M, e = 0.0, a = 1.0, N = 2.0, n = 250, 
                        simulation = "Relativistic")
            
        with pytest.raises(ValueError, match="Eccentricity should take a value in the range \\[0\\., 1\\.\\)"):
            orbit.RunOrbits(M = 5.e6, e = e, a = 1.0, N = 2.0, n = 250, 
                            simulation = "Relativistic")
            
        with pytest.raises(ValueError, match="Semi-major axis should be larger than 0"):
            orbit.RunOrbits(M = 5.e6, e = 0.0, a = a, N = 2.0, n = 250, 
                            simulation = "Relativistic")
            
        with pytest.raises(ValueError, match="The number of orbit periods should be larger than 0"):
            orbit.RunOrbits(M = 5.e6, e = 0.0, a = 1.0, N = N, n = 250, 
                            simulation = "Relativistic")

        with pytest.raises(ValueError, match="The time step number should be at least 50"):
            orbit.RunOrbits(M = 5.e6, e = 0.0, a = 1.0, N = 2.0, n = n, 
                            simulation = "Relativistic") 
        
    def test_input_values(self):
        """
        Unit test to validate that all input attributes of the orbit instances 
        fall within acceptable physical ranges.
        """
        instances = [self.orbit1, self.orbit2] # Save instances in a list

        # Black hole mass can not be cero or negative 
        assert instances[0].M > 0., "Black hole mass can not be cero or negative"
        assert instances[1].M > 0., "Black hole mass can not be cero or negative"

        # Eccentricity should take a value in the range [0., 1.)
        assert instances[0].e >= 0. and instances[0].e < 1., "Eccentricity should take a value in the range [0., 1.)"
        assert instances[1].e >= 0. and instances[1].e < 1., "Eccentricity should take a value in the range [0., 1.)"

        # Semi-amjor axis should be larger than 0 
        assert instances[0].a > 0., "Semi-amjor axis should be larger than 0"
        assert instances[1].a > 0., "Semi-amjor axis should be larger than 0"

        # The number of orbit periods should be larger than 0
        assert instances[0].N > 0., "The number of orbit periods should be larger than 0"
        assert instances[1].N > 0., "The number of orbit periods should be larger than 0"

        # The time step number should be at least 50 
        assert instances[0].n > 50., "The time step number should be at least 50"
        assert instances[1].n > 50., "The time step number should be at least 50" 


    def test_integrator_method(self):
        """
        Verifies that invalid or empty integration methods raise the correct exceptions.
        """

        method = "method_invalid"  # Intentionally invalid

        with pytest.raises(ValueError, match=f" {method} is not available"):
            self.orbit1.solve_ODE(method)

        with pytest.raises(ValueError, match=f"{method} is not available"):
            self.orbit2.solve_ODE(method)

        method = "" #Empty Method
        
        with pytest.raises(ValueError, match="Specify a method"):
            self.orbit2.solve_ODE(method)


    def test_outputs(self):
        """
        Confirms that different orbital parameters result in different numerical solutions.
        Ensures that solutions are not identical when eccentricity and semi-major axis differ.
        """
        _, sol1 = self.orbit1.solve_ODE("RK3")
        _, sol2 = self.orbit2.solve_ODE("RK3")

        print(np.allclose(sol1, sol2, rtol=1e-8, atol=1e-8))

        assert not np.allclose(sol1, sol2, rtol=1e-8, atol=1e-8), "Solutions should differ for different eccentricities"

    def test_initial_conditions(self):
        """
        Validates the computed initial conditions of both orbit instances.
        Checks position and velocity vectors to ensure they match theoretical expectations.
        """
        
        # Initial conditions of orbit1 instance
        x0_1 = self.orbit1.x0 
        y0_1 = self.orbit1.y0  
        vx0_1 = self.orbit1.vx0
        vy0_1 = self.orbit1.vy0

        # Initial conditions of orbit2 instance
        x0_2 = self.orbit2.x0 
        y0_2 = self.orbit2.y0  
        vx0_2 = self.orbit2.vx0
        vy0_2 = self.orbit2.vy0

        # Test intial positions
        assert np.allclose(x0_1, x0_2, rtol=1e-10, atol=1e-10), "Intial contidions x0 should be 0 in every simulation"
        assert not np.allclose(y0_1, y0_2, rtol=1e-10, atol=1e-10), "Intial conditions y0 should be different if major-axis or"\
                                        "eccentricty differ"
        
        # Test initial speed
        assert not np.allclose(vx0_1, vx0_2, rtol=1e-10, atol=1e-10), "Intial conditions y0 should be different if major-axis or"\
                                        "eccentricty differ"
        assert np.allclose(vy0_1, vy0_2, rtol=1e-10, atol=1e-10), "Intial contidions vy0 should be 0 in every simulation"

