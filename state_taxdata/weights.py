

class Cyipopt_functions(object):
    def __init__(self):
        pass

    def objective(self, x, iweight, objscale):
        #
        # The callback for calculating the objective
        #
        w = iweight / objscale
        obj = (w * (x-1)^2).sum()
        return obj

    def gradient(self, x, iweight, objscale):
        #
        # The callback for calculating the gradient
        #
        w = iweight / objscale
        gradf = 2 * w * (x-1)
        return gradf

    def constraints(self, x, ccsparse):
        #
        # The callback for calculating the constraints
        #
        cons = ccsparse.groupby("i")
        return np.array((np.prod(x), np.dot(x, x)))

    def jacobian(self, x):
        #
        # The callback for calculating the Jacobian
        #
        return np.concatenate((np.prod(x) / x, 2*x))

    def hessianstructure(self):
        #
        # The structure of the Hessian
        # Note:
        # The default hessian structure is of a lower triangular matrix. Therefore
        # this function is redundant. I include it as an example for structure
        # callback.
        #
        global hs

        hs = sps.coo_matrix(np.tril(np.ones((4, 4))))
        return (hs.col, hs.row)

    def hessian(self, x, lagrange, obj_factor):
        #
        # The callback for calculating the Hessian
        #
        H = obj_factor*np.array((
                (2*x[3], 0, 0, 0),
                (x[3],   0, 0, 0),
                (x[3],   0, 0, 0),
                (2*x[0]+x[1]+x[2], x[0], x[0], 0)))

        H += lagrange[0]*np.array((
                (0, 0, 0, 0),
                (x[2]*x[3], 0, 0, 0),
                (x[1]*x[3], x[0]*x[3], 0, 0),
                (x[1]*x[2], x[0]*x[2], x[0]*x[1], 0)))

        H += lagrange[1]*2*np.eye(4)

        #
        # Note:
        #
        #
        return H[hs.row, hs.col]

    def intermediate(
            self,
            alg_mod,
            iter_count,
            obj_value,
            inf_pr,
            inf_du,
            mu,
            d_norm,
            regularization_size,
            alpha_du,
            alpha_pr,
            ls_trials
            ):

        #
        # Example for the use of the intermediate callback.
        #
        print "Objective value at iteration #%d is - %g" % (iter_count, obj_value)