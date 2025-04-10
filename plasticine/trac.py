# File from: https://github.com/ComputationalRobotics/TRAC/blob/main/package/src/trac_optimizer/trac.py

from typing import Tuple, Any, Callable, Dict
import torch


# We depend on Erfi function, but torch.special currently has no implementation.
# We instead modify and rely on https://github.com/redsnic/torch_erf

def polyval(x,coeffs):
    """Implementation of the Horner scheme to evaluate a polynomial

    taken from https://discuss.pytorch.org/t/polynomial-evaluation-by-horner-rule/67124

    Args:
        x (torch.Tensor): variable
        coeffs (torch.Tensor): coefficients of the polynomial
    """
    curVal=0
    for curValIndex in range(len(coeffs)-1):
        curVal=(curVal+coeffs[curValIndex])*x[0]
    return(curVal+coeffs[len(coeffs)-1])


class ERF_1994(torch.nn.Module):
    """Class to compute the error function of a complex number (extends torch.special.erf behavior)

    This class is based on the algorithm proposed in:
    Weideman, J. Andre C. "Computation of the complex error function." SIAM Journal on Numerical Analysis 31.5 (1994): 1497-1518
    """
    def __init__(self, n_coefs):
        """Defaul constructor

        Args:
            n_coefs (integer): The number of polynomial coefficients to use in the approximation
        """
        super(ERF_1994, self).__init__()
        # compute polynomial coefficients and other constants
        self.N = n_coefs
        self.i = torch.complex(torch.tensor(0.),torch.tensor(1.))
        self.M = 2*self.N
        self.M2 = 2*self.M
        self.k = torch.linspace(-self.M+1, self.M-1, self.M2-1)
        self.L = torch.sqrt(self.N/torch.sqrt(torch.tensor(2.)))
        self.theta = self.k*torch.pi/self.M
        self.t = self.L*torch.tan(self.theta/2)
        self.f = torch.exp(-self.t**2)*(self.L**2 + self.t**2)
        self.a = torch.fft.fft(torch.fft.fftshift(self.f)).real/self.M2
        self.a = torch.flipud(self.a[1:self.N+1])

    def w_algorithm(self, z):
        """Compute the Faddeeva function of a complex number

        The constant coefficients are computed in the constructor of the class.

        Weideman, J. Andre C. "Computation of the complex error function." SIAM Journal on Numerical Analysis 31.5 (1994): 1497-1518

        Args:
            z (torch.Tensor): A tensor of complex numbers (any shape is allowed)

        Returns:
            torch.Tensor: w(z) for each element of z
        """
        Z = (self.L+self.i*z)/(self.L-self.i*z)
        p = polyval(Z.unsqueeze(0), self.a)
        w = 2*p/(self.L-self.i*z)**2+(1/torch.sqrt(torch.tensor(torch.pi)))/(self.L-self.i*z)
        return w

    def forward(self, z):
        """Compute the error function of a complex number

        The result is computed by manipulating the Faddeeva function.

        Args:
            z (torch.Tensor): A tensor of complex numbers (any shape is allowed)

        Returns:
            torch.Tensor: erf(z) for each element of z
        """
        # exploit the symmetry of the error function
        # find the sign of the real part
        sign_r = torch.sign(z.real)
        sign_i = torch.sign(z.imag)
        # flip sign of imaginary part if negative
        z = torch.complex(torch.abs(z.real), torch.abs(z.imag))
        out = -torch.exp(torch.log(self.w_algorithm(z*self.i)) - z**2) + 1
        return torch.complex(out.real*sign_r, out.imag*sign_i)

    def backward(self, z):
        """Compute the gradient of the error function of a complex number.

        As we know the analytical derivative of the the error function, we can use it directly.

        Args:
            z (torch.Tensor): A tensor of complex numbers (any shape is allowed)
        Returns:
            torch.Tensor: grad(erf(z)) for each element of x
        """
        return 2/torch.sqrt(torch.tensor(torch.pi))*torch.exp(-z**2)

erf_torch = ERF_1994(128)

def erfi(x):
    if not torch.is_floating_point(x):
        x = x.to(torch.float32)

    # Convert x to a complex tensor where the real part is zero
    ix = torch.complex(torch.zeros_like(x), x)

    # Compute erf(ix) / i
    erfi_x = erf_torch(ix).imag  # Extract the imaginary part of erf(ix)
    return erfi_x

# We closely follow the meta-optimizer structure from the code in
# Cutkosky et. al 2023
def _init_state(
        optimizer: torch.optim.Optimizer,
        theta_ref: Dict[torch.Tensor, torch.Tensor],
        betas: Tuple[float],
        s_prev: float,
        eps: float):
    if '_trac' not in optimizer.state:
        optimizer.state['_trac'] = {
            'betas': torch.tensor(betas),
            's_prev': torch.tensor(s_prev),
            'eps': eps,
            's': torch.zeros(len(betas)),
            'theta_ref': {},
            'variance': torch.zeros(len(betas)),
            'sigma': torch.full((len(betas),), 1e-8),
            'iter_count': 0,
        }
        _init_reference(optimizer, theta_ref)

def _init_reference(
        optimizer: torch.optim.Optimizer,
        theta_ref: Dict[torch.Tensor, torch.Tensor],):
    '''
    Args:
        optimizer: optimizer instance to store reference for.
        theta_ref: mapping of parameters to their initial values at the start of optimization.
    '''
    for group in optimizer.param_groups:
        for p in group['params']:
            optimizer.state['_trac'][p] = {
                'ref': theta_ref[p].clone(),
            }
            

def _step(
        optimizer: torch.optim.Optimizer,
        base_step: Callable,
        betas: Tuple[float],
        s_prev: float,
        eps: float,
        ):
    '''
    Args:
        optimizer: trac optimizer instance
        base_step: The "step" function of the base optimizer
        betas: list of beta values.
        s_init: initial scale value.
        eps: epsilon value.
    '''

    prev_grad = torch.is_grad_enabled()


    torch.set_grad_enabled(False)
    updates = {}
    grads = {}
    deltas = {}

    for group in optimizer.param_groups:
        for p in group['params']:

            if p.grad is None:
                grads[p] = None
            else:
                grads[p] = p.grad.clone()
            updates[p] = p.data.clone()

    torch.set_grad_enabled(prev_grad)
    result = base_step(None)
    torch.set_grad_enabled(False)
    
    _init_state(optimizer, updates, betas, s_prev, eps)
    trac_state = optimizer.state['_trac']


    for group in optimizer.param_groups:
        for p in group['params']:
            if grads[p] is None:
                continue

            theta_ref = trac_state[p]['ref']

            deltas[p] = (updates[p] - theta_ref)/(torch.sum(trac_state['s']) + trac_state['eps'])

            updates[p].copy_(p-updates[p])

    h = 0.0
    for group in optimizer.param_groups:
        for p in group['params']:

            if grads[p] is None:
                continue

            grad = grads[p]

            delta = deltas[p]
            product = torch.dot(delta.flatten(), grad.flatten())
            if product.isnan():
                raise ValueError("NaNs in product")
            h += product

            delta.add_(updates[p])

    device = h.device

    for key in trac_state:
        try:
            if trac_state[key].device != device:
                trac_state[key] = trac_state[key].to(device)
        except:
            pass

    s = trac_state['s']
    s_prev = trac_state['s_prev']
    betas = trac_state['betas']
    eps = trac_state['eps']
    variance = trac_state['variance'] 
    sigma = trac_state['sigma']                                 
    trac_state['iter_count'] += 1

    variance.mul_(
        betas**2).add_(torch.square(h))
    sigma.mul_(betas).sub_(h)
    f_term = s_prev / (erfi(torch.tensor(1.0) / torch.sqrt(torch.tensor(2.0))))
    s_term = erfi(sigma / (torch.sqrt(torch.tensor(2.0)) * torch.sqrt(variance) + eps))
    if (f_term * s_term).isnan().any():
        raise ValueError("NaNs in s")
    s.copy_(f_term * s_term)

    for group in optimizer.param_groups:
        for p in group['params']:

            if grads[p] is None:
                continue

            theta_ref = trac_state[p]['ref']
            delta = deltas[p]
            s_sum = torch.sum(s)

            scale = max(s_sum, 0.0)
            p.copy_(theta_ref + delta * scale)

    log_data = {
        'iter_count': trac_state['iter_count'],
        's': torch.sum(s).item(),
    }

    torch.set_grad_enabled(prev_grad)
    return result, log_data


class trac:
    pass

def is_trac(opt):
    return isinstance(opt, trac)

def start_trac(
        log_file,
        Base: Any,
        betas: Tuple[float] = (0.9, 0.99, 0.999, 0.9999,
                               0.99999, 0.999999),
        s_prev: float = 1e-8,
        eps: float = 1e-8,
        ):

    class TRACOPT(Base, trac):
        '''
        Wraps the base opt with trac.
        
        '''

        def step(self):
            result, log_data = _step(self, super().step, betas, s_prev, eps)
            with open (log_file, 'a') as f:
                f.write(str(log_data) + '\n')
            return result

    TRACOPT.__name__ += Base.__name__

    return TRACOPT