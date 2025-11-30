import torch
from torch import vmap
from torch.func import jacrev   # PyTorch ≥ 2.3, or use functorch if older
import time

# -------------------------------
# Quaternion utilities (batched)
# -------------------------------
def quat_exp_batch(xi):
    """
    Batched quaternion exponential map
    xi: (B,3)
    returns q: (B,4)
    """
    theta = torch.norm(xi, dim=1, keepdim=True).clamp(min=1e-16) # (B,1)
    axis = xi / theta
    w = torch.cos(theta)
    xyz = axis * torch.sin(theta)
    return torch.cat([w, xyz], dim=1)  # (B,4)

def quat_im_batch(q):
    """Batched imaginary part of quaternion"""
    return q[:, 1:]  # (B,3)

def skew_batch(v):
    """Batched skew-symmetric matrices"""
    # v: (B,3)
    B = v.shape[0]
    zero = torch.zeros(B, device=v.device, dtype=v.dtype)
    M = torch.stack([
        torch.stack([zero, -v[:,2], v[:,1]], dim=1),
        torch.stack([v[:,2], zero, -v[:,0]], dim=1),
        torch.stack([-v[:,1], v[:,0], zero], dim=1)
    ], dim=1)  # (B,3,3)
    return M

def G_batch(q):
    """Batched G(q) = qs*I - [qv]_x"""
    qs = q[:, 0:1]  # (B,1)
    qv = q[:, 1:]   # (B,3)
    B = q.shape[0]
    I = torch.eye(3, device=q.device, dtype=q.dtype).unsqueeze(0).repeat(B,1,1)  # (B,3,3)
    return qs[:,:,None]*I - skew_batch(qv)  # (B,3,3)

def RHS_batch(xi, J):
    """Batched RHS(xi) = G(exp(-xi))JIm(exp(-xi))"""
    q = quat_exp_batch(-xi)
    return torch.bmm(torch.bmm(G_batch(q), J), quat_im_batch(q).unsqueeze(-1)).squeeze(-1)


def jacobian_batch(xi, J):
    """
    Analytic Jacobian of F(xi) \approx G(exp(-xi)) J Im(exp(-xi))
    xi: (B,3)
    J: (B,3,3)
    Returns: (B,3,3)
    """
    B = xi.shape[0]
    theta = torch.norm(xi, dim=1, keepdim=True).clamp(min=1e-10)  # (B,1)
    axis = xi / theta  # (B,3)

    # quaternion
    q = quat_exp_batch(-xi)
    # qs = q[:,0:1]  # (B,1)
    qv = q[:,1:]   # (B,3)
    # Im_q = qv  # (B,3)
    
    # # derivatives
    # I3 = torch.eye(3, device=xi.device, dtype=xi.dtype).unsqueeze(0).repeat(B,1,1)
    # skew_qv = skew_batch(qv)
    
    # d(Im(q))/d(xi) using standard formula
    # theta2 = theta**2
    sin_theta = torch.sin(theta)
    cos_theta = torch.cos(theta)
    # coeff1 = -0.5*torch.ones_like(theta)  # factor for simplification
    # coeff2 = ((theta - sin_theta)/theta2).squeeze(-1)

    I3 = torch.eye(3, device=xi.device, dtype=xi.dtype).unsqueeze(0).repeat(B,1,1)  # (B,3,3)
    v_skew = skew_batch(xi)  # (B,3,3)
    outer = xi.unsqueeze(2) @ xi.unsqueeze(1)  # (B,3,3)

    dIm_dxi = (-cos_theta[:,:,None]*I3
            + (sin_theta[:,:,None]/theta[:,:,None])*v_skew
            + ((1 - sin_theta/theta)[:,:,None]/theta[:,:,None]**2)*outer)

    # Jacobian F(xi) = G(q) J d(Im(q))/dxi + d(G(q))/dxi J Im(q)
    # Approximate the second term as negligible
    JF = torch.bmm(G_batch(q), J) @ dIm_dxi  # (B,3,3)
    return JF + skew_batch(xi) @ J @ (-xi).unsqueeze(-1)  # (B,3,1)

# -------------------------------
# Batched Newton solver
# -------------------------------
def solve_xi_batch(C, J_theta, xi0=None, tol=1e-15, max_iter=100):
    """
    Solve G(exp(-xi)) J_theta Im(exp(-xi)) = C for xi in batch.
    C: (B,3)
    J_theta: (B,3,3)
    xi0: (B,3) initial guess
    """
    B = C.shape[0]
    device = C.device
    dtype = C.dtype

    if xi0 is None:
        xi = torch.zeros(B, 3, device=device, dtype=dtype, requires_grad=True)
    else:
        xi = xi0.clone().detach().requires_grad_(True)

    for i in range(max_iter):
        F = RHS_batch(xi, J_theta) - C # residual

        normF = torch.norm(F, dim=1).max().item()
        if normF < tol:
            print(f"Converged in {i} iterations, max |F| = {normF:.3e}")
            return xi

        # Compute batched Jacobian using autograd
        J = torch.zeros(B, 3, 3, device=device, dtype=dtype)
        for j in range(3):
            grad = torch.autograd.grad(F[:, j].sum(), xi, retain_graph=True, create_graph=True)[0]
            J[:, j, :] = grad

        # Newton step
        dF_inv = torch.inverse(J)  # (B,3,3)

        xi = xi - torch.bmm(dF_inv, F.unsqueeze(-1)).squeeze(-1)  # (B,3)

        print(f"({i}) max |F| = {normF:.3e}")

    raise RuntimeError(f"Damped Newton did not converge, max |F| = {normF:.3e}")

# -------------------------------
# Test script
# -------------------------------
if __name__ == "__main__":
    from LieFVIN import PSD
    
    B = 10000 # batch size
    dtype = torch.float64
    device = torch.device(0)  # or 'cuda' for GPU

    # torch.manual_seed(0)

    # Example J_theta and C
    A = torch.randn(3,3,dtype=dtype, device=device)
    J_theta_map = PSD(3, 10, 3, init_gain=1.0).to(device).to(dtype)
    J_theta = J_theta_map(torch.randn(B, 3, dtype=dtype, device=device))  # (B,3,3)
    gt_xi = 0.1 * torch.randn(B, 3, dtype=dtype, device=device)
    C = RHS_batch(gt_xi.clone(), J_theta.clone())

    # Warm-up
    xi_batch = solve_xi_batch(C.clone(), J_theta.clone())

    # Compute
    # Time the batched solve (with CUDA synchronization for accurate timing)
    torch.cuda.synchronize()
    t0 = time.time()

    xi_batch = solve_xi_batch(C.clone(), J_theta.clone())

    torch.cuda.synchronize()
    t1 = time.time()

    print(f"Solve time: {t1 - t0:.6f} s")

    # Verify residuals
    F_check = RHS_batch(xi_batch, J_theta)
    F_check = F_check.squeeze(-1) - C
    print("Max residual =", F_check.abs().max().item())
