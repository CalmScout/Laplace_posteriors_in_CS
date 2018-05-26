from constants import N, M, BETTA_INIT, GAMMA_INIT


class Data(object):
    """
    Keeps data for system generation and solving.
    """
    def __init__(self, n=N, m=M, beta_init=BETTA_INIT, gamma_init=GAMMA_INIT):
        self.n = n
        self.m = m
        self.beta = beta_init
        self.gamma = gamma_init
        self.Phi = None
        self.w = None   # values we will approximate
        self.y = None
        self.mu = None
        self.delta = None
        self.z = None
        self.systems_params = None

    def __str__(self):
        result = "Object of 'Data' class.\n"
        result += "n == {}\n".format(self.n)
        result += "m == {}\n".format(self.m)
        result += "beta == {}\n".format(self.beta)
        result += "gamma == {}\n".format(self.gamma)
        result += "Phi == {}\n".format(self.Phi)
        result += "w == {}\n".format(self.w)
        result += "y == {}\n".format(self.y)
        result += "mu == {}\n".format(self.mu)
        result += "delta == {}\n".format(self.delta)
        result += "z == {}\n".format(self.z)
        result += "systems_params == {}\n".format(self.systems_params)
        return result
