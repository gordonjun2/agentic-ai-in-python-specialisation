
import cmath

def quadratic_formula(a, b, c):
    """
    Solves the quadratic equation ax^2 + bx + c = 0 and returns the roots.

    The quadratic formula is used to find the values of x that satisfy the
    equation ax^2 + bx + c = 0. The formula is given by:
        x = (-b ± √(b² - 4ac)) / (2a)

    This function returns both solutions (roots) regardless of whether they
    are real or complex numbers. Complex numbers will be returned if the 
    discriminant (b² - 4ac) is negative, which indicates the roots are 
    imaginary.

    Parameters:
    a (float): Coefficient of x^2. Should not be zero. If zero, raises ValueError.
    b (float): Coefficient of x.
    c (float): Constant term.

    Returns:
    tuple: A tuple containing both roots, which may be real or complex numbers.

    Raises:
    ValueError: If 'a' is zero since the equation would not be quadratic.

    Examples:
    ---------
    1. Finding roots for the equation x^2 - 3x + 2 = 0:
        roots = quadratic_formula(1, -3, 2)
        # roots would be (2.0, 1.0)

    2. Finding roots for an equation with complex roots:
        roots = quadratic_formula(1, 2, 5)
        # roots would be (-1+2j, -1-2j)

    Edge Cases:
    -----------
    - If a = 0, the function raises a ValueError since the equation is not quadratic.
    - If b² = 4ac, the discriminant is zero, indicating a repeated real root.
    - If b² < 4ac, the roots will be complex numbers.
    """
    if a == 0:
        raise ValueError("Coefficient 'a' must not be zero for a quadratic equation.")
    
    # Calculate the discriminant
    discriminant = b**2 - 4*a*c

    # Calculate two roots using the quadratic formula
    root1 = (-b + cmath.sqrt(discriminant)) / (2*a)
    root2 = (-b - cmath.sqrt(discriminant)) / (2*a)

    return (root1, root2)

# Example usage:
# roots = quadratic_formula(1, -3, 2)
# print("Roots:", roots)


import unittest
import cmath


class TestQuadraticFormula(unittest.TestCase):

    def setUp(self):
        """Set up any necessary data for tests."""
        pass

    def test_basic_functionality(self):
        """Test basic cases with real roots."""
        roots = quadratic_formula(1, -3, 2)
        self.assertEqual(roots, (2.0, 1.0))

    def test_complex_roots(self):
        """Test cases where the roots should be complex."""
        roots = quadratic_formula(1, 2, 5)
        self.assertEqual(roots, (complex(-1, 2), complex(-1, -2)))

    def test_repeated_real_root(self):
        """Test cases where the discriminant is zero, leading to repeated roots."""
        roots = quadratic_formula(1, 2, 1)
        expected_root = -1.0
        self.assertEqual(roots, (expected_root, expected_root))

    def test_zero_coefficient_a(self):
        """Test case where 'a' is zero, which should raise a ValueError."""
        with self.assertRaises(ValueError):
            quadratic_formula(0, 2, 1)

    def test_negative_coefficient(self):
        """Test cases with negative coefficients."""
        roots = quadratic_formula(-1, -3, -2)
        self.assertEqual(roots, (-2.0, 1.0))

    def test_zero_coefficient_b(self):
        """Test case where 'b' is zero."""
        roots = quadratic_formula(1, 0, -4)
        self.assertEqual(roots, (2.0, -2.0))

    def test_zero_coefficient_c(self):
        """Test case where 'c' is zero."""
        roots = quadratic_formula(1, -2, 0)
        self.assertEqual(roots, (2.0, 0.0))

    def test_large_coefficients(self):
        """Test case with very large coefficients."""
        roots = quadratic_formula(1e6, -3e7, 2e6)
        # The result is sensitive to small numeric errors, so use isclose for comparison
        self.assertTrue(
            cmath.isclose(roots[0], (2.9789696, -0.0210304), rel_tol=1e-7)
        )

    def tearDown(self):
        """Tear down any data after tests."""
        pass


def quadratic_formula(a, b, c):
    """
    Solves the quadratic equation ax^2 + bx + c = 0 and returns the roots.

    The quadratic formula is used to find the values of x that satisfy the
    equation ax^2 + bx + c = 0. The formula is given by:
        x = (-b ± √(b² - 4ac)) / (2a)

    This function returns both solutions (roots) regardless of whether they
    are real or complex numbers. Complex numbers will be returned if the 
    discriminant (b² - 4ac) is negative, which indicates the roots are 
    imaginary.

    Parameters:
    a (float): Coefficient of x^2. Should not be zero. If zero, raises ValueError.
    b (float): Coefficient of x.
    c (float): Constant term.

    Returns:
    tuple: A tuple containing both roots, which may be real or complex numbers.

    Raises:
    ValueError: If 'a' is zero since the equation would not be quadratic.

    Examples:
    ---------
    1. Finding roots for the equation x^2 - 3x + 2 = 0:
        roots = quadratic_formula(1, -3, 2)
        # roots would be (2.0, 1.0)

    2. Finding roots for an equation with complex roots:
        roots = quadratic_formula(1, 2, 5)
        # roots would be (-1+2j, -1-2j)

    Edge Cases:
    -----------
    - If a = 0, the function raises a ValueError since the equation is not quadratic.
    - If b² = 4ac, the discriminant is zero, indicating a repeated real root.
    - If b² < 4ac, the roots will be complex numbers.
    """
    if a == 0:
        raise ValueError("Coefficient 'a' must not be zero for a quadratic equation.")

    # Calculate the discriminant
    discriminant = b ** 2 - 4 * a * c

    # Calculate two roots using the quadratic formula
    root1 = (-b + cmath.sqrt(discriminant)) / (2 * a)
    root2 = (-b - cmath.sqrt(discriminant)) / (2 * a)

    return (root1, root2)


if __name__ == '__main__':