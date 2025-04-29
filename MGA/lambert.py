import numpy as np 

def LambertIzzo(mu : float, pos1 : np.ndarray, pos2 : np.array, tof : float , nrev : int = 0, solverTol : float = 1e-14):
    """
    Description:
        This function estimates and determines the initial orbit using Lambert's method. 
        The implementation follows two separate solvers
            - Dr. D. Izzo version that is extremely fast, but fails for larger number of revs. 
            - Lancaster & Blancard with improvements by R.Gooding which is a lot slower but more robust.
            When the Izzo version fails, the Lancaster version is automatically called.
            Ref 1 :  http://www.esa.int/gsp/ACT/inf/op/globopt.htm
            Ref 2 : Lancaster, E.R. and Blanchard, R.C. "A unified form of Lambert's theorem." NASA technical note TN D-5368,1969.
            Ref 3 : Gooding, R.H. "A procedure for the solution of Lambert's orbital boundary-value problem. Celestial Mechanics and Dynamical Astronomy

    Args:
        pos1      (numpy.ndarray)       : position vector 1 [km]
        pos2      (numpy.ndarray)       : position vector 2 [km]   
        tof       (float)               : Time of flight between measurements [days].
        nrev      (integer)             : number of revolutions. Defaults to 0. 
        solverTol (float)               : Tolerance of the Lambert solver 

    Returns:
        vel1      (numpy.ndarray)       : Velocity at pos1
        vel2      (numpy.ndarray)       : Velocity at pos2
        exitFlag  (int)                 : +1 indicates success | -1 indicates no solution | -2 indicates 
    """

    # Check inputs 
    if not isinstance(pos1, np.ndarray) :
        raise TypeError(f"Invalid position1 argument. Looking for type 'np.ndarray'.")
    if not isinstance(pos2, np.ndarray) :
        raise TypeError(f"Invalid position2 argument. Looking for type 'np.ndarray'.")    
    if not isinstance(nrev, int):
        raise TypeError(f"Invalid nrev argument. Looking for type 'int'.")
    if not isinstance(tof, float):
        raise TypeError(f"Invalid tof argument. Looking for type 'float'.")
    if not isinstance(mu, float):
        raise TypeError(f"Invalid mu argument. Looking for type 'float'.")    
    if not isinstance(solverTol, float):
        raise TypeError(f"Invalid solver tolerance argument. Looking for type 'float'.") 

    # Initialize 
    bad = False

    # work with non-dimensional units
    r1    = np.sqrt(sum(pos1*pos1))
    r1vec = pos1/r1
    V     = np.sqrt(mu/r1)
    r2vec = pos2/r1
    T     = r1/V
    tf    = tof*86400/T
    m     = nrev

    # relevant geometry parameters (non dimensional)
    mr2vec = np.sqrt(sum(r2vec*r2vec))

    # make 100# sure it's in (-1 <= dth <= +1)
    dth = np.arccos(max(-1, min(1, sum(r1vec*r2vec)/mr2vec)))

    # decide whether to use the left or right branch (for multi-revolution problems), and the long or short way
    leftbranch = np.sign(m)
    longway = np.sign(tf)
    m = abs(m)         
    tf = abs(tf)
    if (longway < 0):
        dth = 2*np.pi - dth

    # derived quantities
    c      = np.sqrt(1 + mr2vec**2 - 2*mr2vec*np.cos(dth))   # non-dimensional chord
    s      = (1 + mr2vec + c)/2                             # non-dimensional semi-perimeter
    aMin   = s/2                                            # minimum energy ellipse semi major axis
    Lambda = np.sqrt(mr2vec)*np.cos(dth/2)/s                # lambda parameter (from BATTIN's book)

    # non-dimensional normal vectors
    crossprd = np.array([r1vec[1]*r2vec[2] - r1vec[2]*r2vec[1],
                        r1vec[2]*r2vec[0] - r1vec[0]*r2vec[2],
                        r1vec[0]*r2vec[1] - r1vec[1]*r2vec[0]])
    mcr       = np.sqrt(sum(crossprd*crossprd)) # magnitues
    nrmunit   = crossprd/mcr                        # unit vector

    # Initial Values
    logt = np.log(tf)

    # Single Rev
    if (m == 0):
        # initial values
        inn1 = -0.5233          # first initial guess
        inn2 = +0.5233          # second initial guess
        x1   = np.log(1 + inn1)    # transformed first initial guess
        x2   = np.log(1 + inn2)    # transformed first second guess
        # multiple revolutions (0, 1 or 2 solutions)
        # the returned soltuion depends on the sign of [m]
    else:
        # select initial values
        if (leftbranch < 0):
            inn1 = -0.5234      # first initial guess, left branch
            inn2 = -0.2234      # second initial guess, left branch
        else:   
            inn1 = +0.7234      # first initial guess, right branch
            inn2 = +0.5234      # second initial guess, right branch
        x1 = np.tan(inn1*np.pi/2)     # transformed first initial guess
        x2 = np.tan(inn2*np.pi/2)     # transformed first second guess

    # since (inn1, inn2) < 0, initial estimate is always ellipse
    xx   = np.array([inn1, inn2])  
    aa = aMin/(1 - np.square(xx))
    bbeta = longway * 2*np.arcsin(np.sqrt((s-c)/2/aa))

    # make 100.4# sure it's in (-1 <= xx <= +1)
    aalfa = 2*np.arccos( np.maximum(-1, np.minimum(1, xx)) )

    # evaluate the time of flight via Lagrange expression
    y12  = aa*np.sqrt(aa)*((aalfa - np.sin(aalfa)) - (bbeta-np.sin(bbeta)) + 2*np.pi*m)

    # initial estimates for y
    if m == 0:
        y1 = np.log(y12[0]) - logt
        y2 = np.log(y12[1]) - logt
    else:
        y1 = y12[0] - tf
        y2 = y12[1] - tf

    # Solve for x using Newton-Raphson iterations
    # If m > 0  and there is no solution. Start the other routine in that case
    err, iterations, xnew = np.inf, 0.0, 0.0
    while (err > solverTol):

        # increment number of iterations
        iterations = iterations + 1

        # new x
        xnew = (x1*y2 - y1*x2) / (y2-y1)

        # copy-pasted code (for performance)
        if m == 0:
            x = np.exp(xnew) - 1 
        else:
            x = np.arctan(xnew)*2/np.pi
        a = aMin/(1 - x**2)

        # ellipse
        if (x < 1): 
            beta = longway * 2*np.arcsin(np.sqrt((s-c)/2/a))
            # make 100.4# sure it's in (-1 <= xx <= +1)
            alfa = 2*np.arccos( max(-1, min(1, x)) )
        # hyperbola
        else: 
            alfa = 2*np.arccosh(x)
            beta = longway * 2*np.arcsinh(np.sqrt((s-c)/(-2*a)))

        # evaluate the time of flight via Lagrange expression
        if (a > 0):
            tof = a*np.sqrt(a)*((alfa - np.sin(alfa)) - (beta-np.sin(beta)) + 2*np.pi*m)
        else:
            tof = -a*np.sqrt(-a)*((np.sinh(alfa) - alfa) - (np.sinh(beta) - beta))

        # new value of y
        if m ==0:
            ynew = np.log(tof) - logt 
        else:
            ynew = tof - tf

        # save previous and current values for the next iterarion
        # (prevents getting stuck between two values)
        x1, x2, y1, y2 = x2, xnew, y2, ynew

        # update error
        err = np.absolute(x1 - xnew)
        # escape clause
        if (iterations > 15):
            bad = True 
            break
        
    # If the Newton-Raphson scheme failed, try to solve the problem with LancasterBlanchard method
    if bad:
        # NOTE: use the original, UN-normalized quantities
        vel1, vel2, exitflag = LambertLancasterBlanchard(mu, r1vec*r1, r2vec*r1, longway*tf*T, leftbranch*m, solverTol)
        return vel1, vel2, exitflag

    # convert converged value of x
    if m==0:
        x = np.exp(xnew) - 1
    else:
        x = np.arctan(xnew)*2/np.pi

    # The solution has been evaluated in terms of log(x+1) or tan(x*pi/2), we now need the conic. 
    # As for transfer angles near to pi the Lagrange-coefficients technique goes singular 
    # (dg approaches a zero/zero that is numerically bad) 
    # we here use a different technique for those cases. 
    # When the transfer angle is exactly equal to pi, then the ih unit vector is not determined. 
    # The remaining equations, though, are still valid.
    # Solution for the semi-major axis
    a = aMin/(1-x**2)
    # Calculate psi
    if (x < 1): # ellipse
        beta = longway * 2*np.arcsin(np.sqrt((s-c)/2/a))
        # make 100.4# sure it's in (-1 <= xx <= +1)
        alfa = 2*np.arccos( max(-1, min(1, x)) )
        psi  = (alfa-beta)/2
        eta2 = 2*a*np.sin(psi)**2/s
        eta  = np.sqrt(eta2)
    else:       # hyperbola
        beta = longway * 2*np.arcsinh(np.sqrt((c-s)/2/a))
        alfa = 2*np.arccosh(x)
        psi  = (alfa-beta)/2
        eta2 = -2*a*np.sinh(psi)**2/s
        eta  = np.sqrt(eta2)

    # unit of the normalized normal vector
    ih = longway * nrmunit

    # unit vector for normalized [r2vec]
    r2n = r2vec/mr2vec

    # cross-products
    # don't use cross() (emlmex() would try to compile it, and this way it
    # also does not create any additional overhead)
    crsprd1 = np.array([ih[1]*r1vec[2]-ih[2]*r1vec[1],
                        ih[2]*r1vec[0]-ih[0]*r1vec[2],
                        ih[0]*r1vec[1]-ih[1]*r1vec[0]])
    crsprd2 = np.array([ih[1]*r2n[2]-ih[2]*r2n[1],
                        ih[2]*r2n[0]-ih[0]*r2n[2],
                        ih[0]*r2n[1]-ih[1]*r2n[0]])

    # radial and tangential directions for departure velocity
    Vr1 = 1/eta/np.sqrt(aMin) * (2*Lambda*aMin - Lambda - x*eta)
    Vt1 = np.sqrt(mr2vec/aMin/eta2 * np.sin(dth/2)**2)

    # radial and tangential directions for arrival velocity
    Vt2 = Vt1/mr2vec
    Vr2 = (Vt1 - Vt2)/np.tan(dth/2) - Vr1

    # terminal velocities
    V1 = (Vr1*r1vec + Vt1*crsprd1)*V
    V2 = (Vr2*r2n + Vt2*crsprd2)*V

    # exitflag
    exitflag = 1 # (success)

    return V1, V2, exitflag

def LambertLancasterBlanchard(mu : float, pos1 : np.ndarray, pos2 : np.ndarray, tf : float, nrev : int = 0, solverTol : float = 1e-14):

    """
    Description:
        This function estimates and determines the initial orbit using Lambert's method. 
        The implementation follows Lancaster & Blancard with improvements by R.Gooding which is a lot slower but more robust.

    Args:
        mu        (float)               : Gravitational paraemter of central body [km3/s2]      
        pos1      (numpy.ndarray)       : position vector 1 [km]
        pos2      (numpy.ndarray)       : position vector 2 [km]   
        tf        (float)               : Time of flight between measurements [days]. 
        nrev      (integer)             : number of revolutions. Defaults to 0. 
        solverTol (float)               : Tolerance of the Lambert solver 

    Returns:
        vel1      (numpy.ndarray)       : Velocity at pos1
        vel2      (numpy.ndarray)       : Velocity at pos2
        exitFlag  (int)                 : +1 indicates success | -1 indicates no solution | -2 indicates 
    """

    # Check inputs 
    if not isinstance(pos1, np.ndarray) :
        raise TypeError(f"Invalid position1 argument. Looking for type 'np.ndarray'.")
    if not isinstance(pos2, np.ndarray) :
        raise TypeError(f"Invalid position2 argument. Looking for type 'np.ndarray'.")
    if not isinstance(nrev, int):
        raise TypeError(f"Invalid nrev argument. Looking for type 'int'.")
    if not isinstance(tf, float):
        raise TypeError(f"Invalid tof argument. Looking for type 'float'.")
    if not isinstance(mu, float):
        raise TypeError(f"Invalid mu argument. Looking for type 'float'.")    
    if not isinstance(solverTol, float):
        raise TypeError(f"Invalid solver tolerance argument. Looking for type 'float'.") 

    # Initialize
    m       = nrev
    r1      = np.sqrt(sum(pos1*pos1))              # magnitude of r1vec
    r2      = np.sqrt(sum(pos2*pos2))              # magnitude of r2vec
    r1unit  = pos1/r1                               # unit vector of r1vec
    r2unit  = pos2/r2                               # unit vector of r2vec
    crsprod = np.cross(pos1,pos2)                  # cross product of r1vec and r2vec
    mcrsprd = np.sqrt(sum(crsprod*crsprod))          # magnitude of that cross product
    th1unit = np.cross(crsprod/mcrsprd,r1unit)       # unit vectors in the tangential-directions
    th2unit = np.cross(crsprod/mcrsprd,r2unit)

    # make 100.4# sure it's in (-1 <= x <= +1)
    dth = np.arccos(np.maximum(-1, np.minimum(1, sum(pos1*pos2)/r1/r2)) )  # turn angle
    # if the long way was selected, the turn-angle must be negative
    # to take care of the direction of final velocity
    longway = np.sign(tf)
    tf = np.absolute(tf)
    if longway < 0:
        dth = dth-2*np.pi

    # left-branch
    leftbranch = np.sign(m)
    m = np.absolute(m)

    # define constants
    c  = np.sqrt(r1**2 + r2**2 - 2*r1*r2*np.cos(dth))
    s  = (r1 + r2 + c) / 2
    T  = np.sqrt(8*mu/s**3) * tf
    q  = np.sqrt(r1*r2)/s * np.cos(dth/2)

    # general formulae for the initial values (Gooding)
    T0  = LancasterBlanchard(0, q, m)[0]
    Td  = T0 - T
    phr = np.mod(2*np.arctan2(1 - q**2, 2*q), 2*np.pi)

    # initial output is pessimistic
    V1 = np.array([np.inf, np.inf, np.inf])
    V2 = V1

    # single-revolution case
    if m == 0:
        x01 = T0*Td/4/T
        if Td > 0:
            x0 = x01
        else:
            x01 = Td/(4 - Td)
            x02 = -np.sqrt( -Td/(T+T0/2) )
            W   = x01 + 1.7*np.sqrt(2 - phr/np.pi)
            if W >= 0:
                x03 = x01
            else:
                x03 = x01 -W**(1/16) *(x02 - x01)
            lambd = 1 + x03*(1 + x01)/2 - 0.03* x03**2 *np.sqrt(1 + x01)
            x0 = lambd*x03
        # this estimate might not give a solution
        if (x0 < -1):
            exitflag = -1 
            return None, None, exitflag

    # multi-revolution case
    else:
        # determine minimum Tp(x)
        xMpi = 4/(3*np.pi*(2*m + 1))
        if (phr < np.pi):
            xM0 = xMpi*(phr/np.pi)**(1/8)
        elif (phr > np.pi):
            xM0 = xMpi*(2 - (2 - phr/np.pi)**(1/8))

        # EMLMEX requires this one
        else:
            xM0 = 0

        # use Halley's method
        xM = xM0  
        Tp = np.inf  
        iterations = 0
        while abs(Tp) > solverTol:
            # iterations
            iterations = iterations + 1
            # compute first three derivatives
            dummy, Tp, Tpp, Tppp = LancasterBlanchard(xM, q, m)
            # new value of xM
            xMp = xM
            xM  = xM - 2*Tp*Tpp / (2*Tpp**2 - Tp*Tppp)
            # escape clause
            if np.mod(iterations, 7):
                xM = (xMp+xM)/2 
            # the method might fail. Exit in that case
            if (iterations > 25):
                exitflag = -2 
                return None, None, exitflag

        # xM should be elliptic (-1 < x < 1)
        # (this should be impossible to go wrong)
        if (xM < -1) or (xM > 1):
            exitflag = -1 
            return None, None, exitflag

        # corresponding time
        TM = LancasterBlanchard(xM, q, m)
        # T should lie above the minimum T
        if (TM > T):
            exitflag = -1 
            return None, None, exitflag

        # find two initial values for second solution (again with lambda-type patch)
        # --------------------------------------------------------------------------
        # some initial values
        TmTM  = T - TM
        T0mTM = T0 - TM
        dummy, Tp, Tpp, notReq = LancasterBlanchard(xM, q, m)##ok

        # first estimate (only if m > 0)
        if leftbranch > 0:
            x   = np.sqrt( TmTM / (Tpp/2 + TmTM/(1-xM)**2) )
            W   = xM + x
            W   = 4*W/(4 + TmTM) + (1 - W)**2
            x0  = x*(1 - (1 + m + (dth - 1/2)) / (1 + 0.15*m)*x*(W/2 + 0.03*x*np.sqrt(W))) + xM
            # first estimate might not be able to yield possible solution
            if (x0 > 1): 
                exitflag = -1 
                return None, None, exitflag 

        # second estimate (only if m > 0)
        else:
            if (Td > 0):
                x0 = xM - np.sqrt(TM/(Tpp/2 - TmTM*(Tpp/2/T0mTM - 1/xM**2)))
            else:
                x00 = Td / (4 - Td)
                W = x00 + 1.7*np.sqrt(2*(1 - phr))
                if (W >= 0):
                    x03 = x00
                else:
                    x03 = x00 - np.sqrt((-W)**(1/8))*(x00 + np.sqrt(-Td/(1.5*T0 - Td)))
                W      = 4/(4 - Td)
                lambd  = (1 + (1 + m + 0.24*(dth - 1/2)) / (1 + 0.15*m)*x03*(W/2 - 0.03*x03*np.sqrt(W)))
                x0     = x03*lambd
            # estimate might not give solutions
            if (x0 < -1): 
                exitflag = -1
                return None, None, exitflag 

    # find root of Lancaster & Blancard's function
    # --------------------------------------------
    # (Halley's method)
    x = x0
    Tx = np.inf
    iterations = 0
    while abs(Tx) > solverTol:
        # iterations
        iterations = iterations + 1
        # compute function value, and first two derivatives
        Tx, Tp, Tpp, notReq = LancasterBlanchard(x, q, m)
        # find the root of the *difference* between the
        # function value [T_x] and the required time [T]
        Tx = Tx - T
        # new value of x
        xp = x
        x  = x - 2*Tx*Tp / (2*Tp**2 - Tx*Tpp)
        # escape clause
        if np.mod(iterations, 7):
            x = (xp+x)/2
        # Halley's method might fail
        if iterations > 25:
            exitflag = -2
            return None, None, exitflag

    # calculate terminal velocities
    # -----------------------------
    # constants required for this calculation
    gamma = np.sqrt(mu*s/2)
    if (c == 0):
        sigma = 1
        rho   = 0
        z     = abs(x)
    else:
        sigma = 2*np.sqrt(r1*r2/(c**2)) * np.sin(dth/2)
        rho   = (r1 - r2)/c
        z     = np.sqrt(1 + q**2*(x**2 - 1))

    # radial component
    Vr1    = +gamma*((q*z - x) - rho*(q*z + x)) / r1
    Vr1vec = Vr1*r1unit
    Vr2    = -gamma*((q*z - x) + rho*(q*z + x)) / r2
    Vr2vec = Vr2*r2unit

    # tangential component
    Vtan1    = sigma * gamma * (z + q*x) / r1
    Vtan1vec = Vtan1 * th1unit
    Vtan2    = sigma * gamma * (z + q*x) / r2
    Vtan2vec = Vtan2 * th2unit

    # Cartesian velocity
    V1 = Vtan1vec + Vr1vec
    V2 = Vtan2vec + Vr2vec

    # exitflag
    exitflag = 1 # (success)

    return V1, V2, exitflag

# Lancaster & Blanchard's function, and three derivatives thereof
def LancasterBlanchard(x, q, m):

    # offset
    eps = 1e-3
    
    # protection against idiotic input
    if (x < -1): # impossible negative eccentricity
        x = abs(x) - 2
    elif (x == -1): # impossible offset x slightly
        x = x + eps
    
    # compute parameter E
    E  = x*x - 1
    
    # T(x), T'(x), T''(x)
    if x == 1: # exactly parabolic solutions known exactly
        
        # T(x)
        T = 4/3*(1-q**3)
        
        # T'(x)
        Tp = 4/5*(q**5 - 1)
        
        # T''(x)
        Tpp = Tp + 120/70*(1 - q**7)
        
        # T'''(x)
        Tppp = 3*(Tpp - Tp) + 2400/1080*(q**9 - 1)
    
    elif abs(x-1) < 1e-2: # near-parabolic compute with series
        
        # evaluate sigma
        sig1, dsigdx1, d2sigdx21, d3sigdx31 = sigmax(-E)
        sig2, dsigdx2, d2sigdx22, d3sigdx32 = sigmax(-E*q*q)
        
        # T(x)
        T = sig1 - q**3*sig2
        
        # T'(x)
        Tp = 2*x*(q**5*dsigdx2 - dsigdx1)
        
        # T''(x)
        Tpp = Tp/x + 4*x**2*(d2sigdx21 - q**7*d2sigdx22)
        
        # T'''(x)
        Tppp = 3*(Tpp-Tp/x)/x + 8*x*x*(q**9*d3sigdx32 - d3sigdx31)
    else: # all other cases
        
        # compute all substitution functions
        y  = np.sqrt(abs(E))
        z  = np.sqrt(1 + q**2*E)
        f  = y*(z - q*x)
        g  = x*z - q*E
        
        # BUGFIX: (Simon Tardivel) this line is incorrect for E==0 and f+g==0
        # d  = (E < 0)*(atan2(f, g) + pi*m) + (E > 0)*log( max(0, f + g) )
        # it should be written out like so:
        if (E<0):
            d = np.arctan2(f, g) + np.pi*m
        elif (E==0):
            d = 0
        else:
            d = np.log(max(0, f+g))
        
        # T(x)
        T = 2*(x - q*z - d/y)/E
        
        #  T'(x)
        Tp = (4 - 4*q**3*x/z - 3*x*T)/E
        
        # T''(x)
        Tpp = (-4*q**3/z * (1 - q**2*x**2/z**2) - 3*T - 3*x*Tp)/E
        
        # T'''(x)
        Tppp = (4*q**3/z**2*((1 - q**2*x**2/z**2) + 2*q**2*x/z**2*(z - x)) - 8*Tp - 7*x*Tpp)/E
    
    return T, Tp, Tpp, Tppp
    
# series approximation to T(x) and its derivatives (used for near-parabolic cases)
def sigmax(y):
    an = np.array([4.000000000000000e-001,     2.142857142857143e-001,     4.629629629629630e-002,
                   6.628787878787879e-003,     7.211538461538461e-004,     6.365740740740740e-005,
                   4.741479925303455e-006,     3.059406328320802e-007,     1.742836409255060e-008,
                   8.892477331109578e-010,     4.110111531986532e-011,     1.736709384841458e-012,
                   6.759767240041426e-014,     2.439123386614026e-015,     8.203411614538007e-017,
                   2.583771576869575e-018,     7.652331327976716e-020,     2.138860629743989e-021,
                   5.659959451165552e-023,     1.422104833817366e-024,     3.401398483272306e-026,
                   7.762544304774155e-028,     1.693916882090479e-029,     3.541295006766860e-031,
                   7.105336187804402e-033])
    
    # range values
    rng1 = range(1,26,1)
    rng2 = range(0,25,1)
    rng3 = range(-1,24,1)
    
    # powers of y
    powers = y**rng1
    
    # sigma itself
    sig = 4/3 + np.matmul(powers,an)
    
    # dsigma / dx (derivative)
    dsigdx = np.matmul(rng1*np.insert(powers[0:24], 0, 1., axis=0), an)
    
    # d2sigma / dx2 (second derivative)
    d2sigdx2 = np.matmul( np.insert(powers[0:23], 0, [1/y, 1.], axis=0)*rng1*rng2, an)

    # d3sigma / dx3 (third derivative)
    d3sigdx3 = np.matmul( np.insert(powers[0:22], 0, [1/y/y, 1/y, 1.], axis=0)*rng1*rng2*rng3, an)

    return sig, dsigdx, d2sigdx2, d3sigdx3