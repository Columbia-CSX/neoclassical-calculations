from simsopt.field.boozermagneticfield import BoozerRadialInterpolant
from simsopt.mhd.vmec import Vmec
import numpy as np

def getBoozerRadialInterpolant(wout_file):
    return BoozerRadialInterpolant(Vmec(wout_file), 3, mpol=15, ntor=14)

def unrollMeshgrid(X, Y, Z):
    """Returns an array of points (N, 3) based off of a meshed grid"""
    return np.ascontiguousarray(np.vstack((X.flatten(), Y.flatten(), Z.flatten())).T, dtype=np.float64)

def rollMeshgrid(len_xcoords, len_ycoords, points):
    """ Takes the length of the x and y coordinate
    array and a list of points shape (N, 3) and returns
    (X, Y, Z) the meshed grids corresponding to the points
    """
    try:
        _, s = points.shape
    except:
        s = 1
    return np.squeeze(points.T.reshape((s, len_xcoords, len_ycoords)))

def cylindrical(bri, points):
    """
    Takes in bri (BoozerRadialInterpolant) and a list of points in
    Boozer coordinates (psiN, theta, zeta), returns the value of these
    points in cylindrical coordinates (R, nu, Z).
    """
    zeta = points[:,2]
    bri.set_points(points)
    R = bri.R()
    phi = zeta - bri.nu()
    Z = bri.Z()
    cpoints = np.transpose(np.array([R, phi, Z]))[0]
    return cpoints

def cartesian(bri, points):
    """
    Takes in bri (BoozerRadialInterpolant) and a list of points in
    Boozer coordinates (psiN, theta, zeta), returns the value of these
    points in cartesian coordinates (x, y, z).
    """
    zeta = np.atleast_2d(points[:,2]).T
    bri.set_points(points)
    R = bri.R()
    phi = zeta - bri.nu()
    Z = bri.Z()
    x = R*np.cos(phi)
    y = R*np.sin(phi)
    cpoints = np.transpose(np.array([x, y, Z]))[0]
    return cpoints

def getGradientMagnitudes(bri, points):
    """
    Takes in bri (BoozerRadialInterpolant) and a list of points in
    Boozer coordinates (psiN, theta, zeta), return a tuple of the values of the
    magnitude of the contravariant basis vectors (dr/dtheta, dr/dzeta)
    as arrays of length Npoints (points.shape=(Npoints, 3)).
    """
    bri.set_points(points)
    R = bri.R()
    dRdtheta = bri.dRdtheta()
    dRdzeta = bri.dRdzeta()
    dnudtheta = bri.dnudtheta()
    dzetadtheta = bri.iota()**(-1)
    dnudzeta = bri.dnudzeta()
    dZdtheta = bri.dZdtheta()
    dZdzeta = bri.dZdzeta()
    normdrdzeta = np.sqrt(dRdzeta**2+(R-R*dnudzeta)**2+dZdzeta**2)
    normdrdtheta = np.sqrt(dRdtheta**2+(R*dzetadtheta-R*dnudtheta)**2+dZdtheta**2)
    return np.squeeze(normdrdtheta), np.squeeze(normdrdzeta)

def get_drdzeta_dot_drdtheta(bri, points):
    bri.set_points(points)
    R = bri.R()
    dRdtheta = bri.dRdtheta()
    dRdzeta = bri.dRdzeta()
    dnudtheta = bri.dnudtheta()
    dzetadtheta = bri.iota()**(-1)
    dnudzeta = bri.dnudzeta()
    dphidtheta = dzetadtheta - dnudtheta
    dphidzeta = 1.0 - dnudzeta
    dZdtheta = bri.dZdtheta()
    dZdzeta = bri.dZdzeta()
    return np.squeeze(dRdtheta*dRdzeta + R*R*dphidtheta*dphidzeta + dZdtheta*dZdzeta)

if __name__ == '__main__':
    bri = getBoozerRadialInterpolant("/global/homes/m/michaelc/stelloptPlusSfincs/equilibria/wout_csx_ls_4.5_0.5T.nc")
    bri.set_points(np.array([[0.1, 0.1, 0.1], [0.5, 0.5, 0.5]]))
    print(bri.zeta())
