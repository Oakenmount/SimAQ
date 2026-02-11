import numpy as np
import pyvista as pv
from typing import List, Tuple, Optional


class Sphere:
    """
    Represents a sphere in 3D space, with methods for inversion and coordinate transformations.
    """
    def __init__(self, center: np.ndarray, radius: Optional[float] = None, curvature: Optional[float] = None):
        """
        Initialize a Sphere with either a radius or curvature.

        Args:
            center (np.ndarray): The center of the sphere.
            radius (Optional[float]): The radius of the sphere.
            curvature (Optional[float]): The curvature of the sphere.
        """
        if radius is None and curvature is None:
            raise ValueError("Either radius or curvature must be provided.")
        if radius is not None:
            self.radius = radius
            self.curvature = 1 / radius
        else:
            self.curvature = curvature  # type: ignore
            self.radius = 1 / curvature  # type: ignore
        self.center = center
        self.inv = self.get_inverse_coordinates(center, self.radius)

    def __repr__(self) -> str:
        return f"Sphere(radius={self.radius}, center={self.center})"

    @staticmethod
    def get_inverse_coordinates(center: np.ndarray, radius: float) -> np.ndarray:
        """
        Compute the inverse coordinates for a sphere.

        Args:
            center (np.ndarray): The center of the sphere.
            radius (float): The radius of the sphere.

        Returns:
            np.ndarray: The inverse coordinates.
        """
        inv = [x / radius for x in center]
        inv.append((np.sum(np.square(center)) - radius ** 2 - 1) / (2 * radius))
        inv.append((np.sum(np.square(center)) - radius ** 2 + 1) / (2 * radius))
        return np.array(inv)

    @staticmethod
    def get_regular_coordinates(inv: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Convert inverse coordinates to regular (center, radius) coordinates.

        Args:
            inv (np.ndarray): The inverse coordinates.

        Returns:
            Tuple[np.ndarray, float]: The center and radius.
        """
        kappa = inv[4] - inv[3]
        r = 1 / kappa
        center = inv[:3] * r
        return center, r

    @staticmethod
    def sphere_from_inverse(inv: np.ndarray) -> "Sphere":
        """
        Create a Sphere from inverse coordinates.

        Args:
            inv (np.ndarray): The inverse coordinates.

        Returns:
            Sphere: The corresponding Sphere object.
        """
        center, radius = Sphere.get_regular_coordinates(inv)
        return Sphere(center, radius)

    @staticmethod
    def special_to_cartesian(inv: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Convert special coordinates to cartesian (center, radius).

        Args:
            inv (np.ndarray): The special coordinates.

        Returns:
            Tuple[np.ndarray, float]: The center and radius.
        """
        rt2o2 = np.sqrt(2) / 2
        rt6o2 = np.sqrt(6) / 2
        a, b, c, d, e = inv
        return Sphere.get_regular_coordinates(np.array([
            (-b + c + d - e) * rt2o2,
            (b - c + d - e) * rt2o2,
            (b + c - d - e) * rt2o2,
            a - b - c - d - e,
            (b + c + d + e) * rt6o2
        ]))


def I(i: int, A: float, B: float, C: float, D: float, E: float) -> List[float]:
    """
    Linear mapping for sphere inversion.

    Args:
        i (int): Index of inversion.
        A, B, C, D, E (float): Sphere parameters.

    Returns:
        List[float]: The mapped parameters.
    """
    if i == 0:
        return [-A, A + B, A + C, A + D, A + E]
    elif i == 1:
        return [B + A, -B, B + C, B + D, B + E]
    elif i == 2:
        return [C + A, C + B, -C, C + D, C + E]
    elif i == 3:
        return [D + A, D + B, D + C, -D, D + E]
    elif i == 4:
        return [E + A, E + B, E + C, E + D, -E]
    else:
        raise ValueError("Index i must be in [0, 4]")


def generate(basis: np.ndarray) -> List[List[float]]:
    """
    Generate new spheres by applying inversion and removing duplicates.

    Args:
        basis (np.ndarray): The current basis of spheres.

    Returns:
        List[List[float]]: The new basis after inversion.
    """
    result = []
    rt6o2 = np.sqrt(6) / 2
    for j in range(len(basis)):
        A, B, C, D, E = basis[j]
        curvJ = (B + C + D + E) * rt6o2 - (A - B - C - D - E)
        for i in range(5):
            tmp = I(i, A, B, C, D, E)
            A1, B1, C1, D1, E1 = tmp
            curvI = (B1 + C1 + D1 + E1) * rt6o2 - (A1 - B1 - C1 - D1 - E1)
            if (
                curvI <= curvJ or
                (i == 0 and (B1 < A1 or C1 < A1 or D1 < A1 or E1 < A1)) or
                (i == 1 and (A1 <= B1 or C1 < B1 or D1 < B1 or E1 < B1)) or
                (i == 2 and (A1 <= C1 or B1 <= C1 or D1 < C1 or E1 < C1)) or
                (i == 3 and (A1 <= D1 or B1 <= D1 or C1 <= D1 or E1 < D1)) or
                (i == 4 and (A1 <= E1 or B1 <= E1 or C1 <= E1 or D1 <= E1))
            ):
                continue
            result.append([A1, B1, C1, D1, E1])
    return result


def generate_apollonian(depth: int = 3) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate centers and radii of spheres in an Apollonian packing up to a given depth.

    Args:
        depth (int): The recursion depth.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Arrays of centers and radii.
    """
    spheres: List[Sphere] = []
    basis = np.identity(5)
    for arr in basis:
        center, radius = Sphere.special_to_cartesian(arr)
        spheres.append(Sphere(center, radius))
    for _ in range(depth):
        basis = generate(basis)
        for arr in basis:
            center, radius = Sphere.special_to_cartesian(arr)
            spheres.append(Sphere(center, radius))
    centers = np.array([sphere.center for sphere in spheres if sphere.radius > 0])
    radii = np.array([sphere.radius for sphere in spheres if sphere.radius > 0])
    return centers, radii


if __name__ == "__main__":
    """
    Visualize the Apollonian packing using pyvista.
    """
    basis = np.identity(5)
    plotter = pv.Plotter()
    colors = ["red", "green", "blue", "yellow", "purple", "orange"]
    for arr in basis:
        center, radius = Sphere.special_to_cartesian(arr)
        sphere = pv.Sphere(radius=radius, center=center)
        plotter.add_mesh(sphere, color=colors[0], opacity=1)

    for i in range(5):
        basis = generate(basis)
        for arr in basis:
            center, radius = Sphere.special_to_cartesian(arr)
            if np.random.rand() > 0.3:
                sphere = pv.Sphere(radius=radius, center=center)
                plotter.add_mesh(sphere, color=colors[i + 1], opacity=1)
    plotter.show()
