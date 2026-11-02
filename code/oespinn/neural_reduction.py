"""Learned spatial trial spaces with the common incremental constitutive law."""
import torch


def hidden_coefficients(model, *, all_layers=False):
    """Project hidden features using the same transform as the displacement field."""
    values = model.inputs
    features = []
    for layer in model.kinematic_trunk:
        values = layer(values)
        if isinstance(layer, torch.nn.Tanh):
            features.append(values)
    if all_layers:
        values = torch.cat(features, dim=1)
    values = values.reshape(model.radial_order+1, model.angular_samples, -1)
    spectrum = torch.fft.rfft(values, dim=1)[:, :model.angular_modes+1]/model.angular_samples
    angular = torch.cat((spectrum[:, :1].real, 2*spectrum[:, 1:].real,
                         -2*spectrum[:, 1:].imag), 1)
    coefficients = torch.einsum('ij,jmc->imc', model.dct, angular)
    return coefficients.flatten(0, 1)[1:]


def vector_features(scalar):
    basis = scalar.new_zeros((2*len(scalar), 2*scalar.shape[1]))
    basis[0::2, 0::2] = scalar
    basis[1::2, 1::2] = scalar
    return basis


@torch.no_grad()
def energy_basis(work, candidates, relative_cutoff=1e-10):
    energy = work.factor.T@candidates
    left, singular, _ = torch.linalg.svd(energy, full_matrices=False)
    keep = singular > singular[0]*relative_cutoff
    basis = torch.linalg.solve_triangular(work.factor.T, left[:, keep], upper=True)
    return basis, singular


class ReducedIncrement:
    """Galerkin balance with unchanged material history and quadrature."""
    def __init__(self, work, basis, prefix, previous_plastic, release):
        self.work, self.basis = work, basis
        self.prefix, self.previous_plastic, self.release = prefix, previous_plastic, release
        x, y = basis[0::2], basis[1::2]
        self.b = torch.stack((work.dx@x, work.dy@y, work.dx@x*0,
                              work.dy@x+work.dx@y), 1)
        self.k = basis.T@work.k@basis
        self.kext = basis.T@work.kext@basis
        self.prefix_strain = work.strain(prefix, (work.value, work.dx, work.dy))
        self.base = basis.T@(work.k@prefix.flatten()-release*work.load)

    def evaluate(self, coordinate, *, tangent=False):
        w = self.work
        strain = self.prefix_strain+torch.einsum('qir,r->qi', self.b, coordinate)
        returned = w.returned(w.s0+(strain-self.previous_plastic)@w.c.T, strict=False)
        plastic = self.previous_plastic+returned['plastic']
        source = (plastic@w.c.T)*w.w[:, None]
        residual = (self.base+self.k@coordinate-torch.einsum('qir,qi->r', self.b, source))/w.load_norm
        if not tangent:
            return residual, returned
        modulus = returned['stress_tangent']@w.c
        db = torch.einsum('qij,qjr->qir', modulus, self.b)
        jacobian = (self.kext+torch.einsum('qir,qis,q->rs', self.b, db, w.w))/w.load_norm
        return residual, jacobian, returned


@torch.no_grad()
def solve_reduced(problem, initial, tolerance=1e-8):
    coordinate = initial.clone()
    records = []
    for iteration in range(80):
        residual, tangent, returned = problem.evaluate(coordinate, tangent=True)
        norm = float(residual.norm())
        if norm <= tolerance:
            return coordinate, returned, records
        merit = .5*norm**2
        normal = None
        for regularization in range(9):
            if regularization == 0:
                direction, info = torch.linalg.solve_ex(tangent, -residual)
                if int(info) != 0:
                    continue
            else:
                if normal is None:
                    normal = tangent.T@tangent
                    gradient = tangent.T@residual
                damping = normal.diag().max()*10.**(regularization-7)
                direction = torch.linalg.solve(normal+damping*torch.eye(len(coordinate),
                    device=coordinate.device, dtype=coordinate.dtype), -gradient)
            slope = float(residual@(tangent@direction))
            if not bool(torch.isfinite(direction).all()) or slope >= 0:
                continue
            alpha = 1.
            for backtrack in range(16):
                try:
                    trial, _ = problem.evaluate(coordinate+alpha*direction)
                    value = float(.5*trial.square().sum())
                except FloatingPointError:
                    value = float('inf')
                if value <= merit+1e-4*alpha*slope:
                    break
                alpha *= .5
            else:
                continue
            break
        else:
            raise RuntimeError(f'Reduced line search failed: {norm}')
        coordinate += alpha*direction
        records.append(dict(iteration=iteration+1, residual=norm, alpha=alpha,
                            trial_residual=(2*value)**.5, backtracks=backtrack))
    raise RuntimeError('Reduced Newton iteration did not converge')
