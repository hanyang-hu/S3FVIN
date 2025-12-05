import torch
import tqdm
import gc

from LieFVIN import MLP, PSD


def quat_mul_batch(q, p):
    """Batched quaternion multiplication q * p"""
    w1, x1, y1, z1 = q[:,0], q[:,1], q[:,2], q[:,3]
    w2, x2, y2, z2 = p[:,0], p[:,1], p[:,2], p[:,3]
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2
    return torch.stack([w, x, y, z], dim=1)  # (B,4)

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

def quat_log_batch(q):
    """
    Batched quaternion logarithm map
    q: (B,4)
    returns xi: (B,3)
    """
    w = q[:, 0:1]  # (B,1)
    v = q[:, 1:]   # (B,3)
    v_norm = torch.norm(v, dim=1, keepdim=True).clamp(min=1e-16)  # (B,1)
    theta = torch.atan2(v_norm, w)  # (B,1)
    axis = v / v_norm  # (B,3)
    xi = axis * (2 * theta)  # (B,3)
    return xi

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
    """Batched G(q) = qs*I - [qv]_x s.t. G(q)^T x = Im(q(0, x))"""
    qs = q[:, 0:1]  # (B,1)
    qv = q[:, 1:]   # (B,3)
    B = q.shape[0]
    I = torch.eye(3, device=q.device, dtype=q.dtype).unsqueeze(0).repeat(B,1,1)  # (B,3,3)
    return qs[:,:,None]*I - skew_batch(qv)  # (B,3,3)

def H_batch(q):
    """Batched H(q) = [qv, G(q)] s.t. H(q)^T x = q(0, x) of shape (B, 3, 4)"""
    qs = q[:, 0:1]  # (B,1)
    qv = q[:, 1:]   # (B,3)
    B = q.shape[0]
    I = torch.eye(3, device=q.device, dtype=q.dtype).unsqueeze(0).repeat(B,1,1)  # (B,3,3)
    Gq = qs[:,:,None]*I - skew_batch(qv)  # (B,3,3)
    Hq = torch.cat([-qv.unsqueeze(-1), Gq], dim=2)  # (B,3,4)
    return Hq

def batch_quat_to_rotmat(q):
    """q: (B,4) in (w,x,y,z) Hamilton convention"""
    w, x, y, z = q[:,0], q[:,1], q[:,2], q[:,3]
    R = torch.zeros(q.shape[0], 3, 3, device=q.device, dtype=q.dtype)
    R[:,0,0] = 1 - 2*(y*y + z*z)
    R[:,0,1] = 2*(x*y - w*z)
    R[:,0,2] = 2*(x*z + w*y)
    R[:,1,0] = 2*(x*y + w*z)
    R[:,1,1] = 1 - 2*(x*x + z*z)
    R[:,1,2] = 2*(y*z - w*x)
    R[:,2,0] = 2*(x*z - w*y)
    R[:,2,1] = 2*(y*z + w*x)
    R[:,2,2] = 1 - 2*(x*x + y*y)
    return R

def batch_rotmat_to_quat(R):
    """Batched rotation matrix to quaternion (w,x,y,z) Hamilton convention"""
    B = R.shape[0]
    q = torch.zeros(B, 4, device=R.device, dtype=R.dtype)

    trace = R[:,0,0] + R[:,1,1] + R[:,2,2]

    # Case 1: trace > 0
    mask = trace > 0
    S = torch.sqrt(trace[mask] + 1.0) * 2  # S=4*w
    q[mask,0] = 0.25 * S
    q[mask,1] = (R[mask,2,1] - R[mask,1,2]) / S
    q[mask,2] = (R[mask,0,2] - R[mask,2,0]) / S
    q[mask,3] = (R[mask,1,0] - R[mask,0,1]) / S

    # Case 2: R[0,0] is the largest diagonal
    mask = (R[:,0,0] > R[:,1,1]) & (R[:,0,0] > R[:,2,2]) & (trace <= 0)
    S = torch.sqrt(1.0 + R[mask,0,0] - R[mask,1,1] - R[mask,2,2]) * 2  # S=4*x
    q[mask,0] = (R[mask,2,1] - R[mask,1,2]) / S
    q[mask,1] = 0.25 * S
    q[mask,2] = (R[mask,0,1] + R[mask,1,0]) / S
    q[mask,3] = (R[mask,0,2] + R[mask,2,0]) / S

    # Case 3: R[1,1] is the largest diagonal
    mask = (R[:,1,1] > R[:,2,2]) & (trace <= 0) & ~( (R[:,0,0] > R[:,1,1]) & (R[:,0,0] > R[:,2,2]) )
    S = torch.sqrt(1.0 + R[mask,1,1] - R[mask,0,0] - R[mask,2,2]) * 2  # S=4*y
    q[mask,0] = (R[mask,0,2] - R[mask,2,0]) / S
    q[mask,1] = (R[mask,0,1] + R[mask,1,0]) / S
    q[mask,2] = 0.25 * S
    q[mask,3] = (R[mask,1,2] + R[mask,2,1]) / S

    # Case 4: R[2,2] is the largest diagonal
    mask = (R[:,2,2] >= R[:,0,0]) & (R[:,2,2] >= R[:,1,1]) & (trace <= 0)
    S = torch.sqrt(1.0 + R[mask,2,2] - R[mask,0,0] - R[mask,1,1]) * 2  # S=4*z
    q[mask,0] = (R[mask,1,0] - R[mask,0,1]) / S
    q[mask,1] = (R[mask,0,2] + R[mask,2,0]) / S
    q[mask,2] = (R[mask,1,2] + R[mask,2,1]) / S
    q[mask,3] = 0.25 * S

    return q

def quat_L2_geodesic_loss(target, target_hat, split):
    """
    Compute L2 and geodesic loss between target and target_hat quaternions
    The input are rotation matrices concatenated with other states.
    """
    q, w, _ = torch.split(target, split, dim=2)
    q_hat, w_hat, _ = torch.split(target_hat, split, dim=2)
    q, q_hat = q.reshape(-1, 4), q_hat.reshape(-1, 4)
    dot = torch.sum(q * q_hat, dim=1, keepdim=True)
    q_hat = torch.where(dot < 0, -q_hat, q_hat)  # Ensure same hemisphere

    l2_loss = (w - w_hat).pow(2).mean()

    q_hat_conj = q_hat * torch.tensor([1, -1, -1, -1], device=q.device, dtype=q.dtype)
    xi = quat_log_batch(quat_mul_batch(q_hat_conj, q))
    geo_loss = torch.norm(xi, dim=1).pow(2).mean()

    total = l2_loss + geo_loss

    return total, l2_loss, geo_loss

class S3FVIN(torch.nn.Module):
    def __init__(self, J_net=None, V_net=None, g_net=None, device=None, u_dim=1, time_step=0.01, init_gain=1, enable_force=True, fixed_J=False, quat_sym=True):
        assert u_dim == 1, "Currently only u_dim=1 is supported."

        super(S3FVIN, self).__init__()

        self.rotmatdim = 9
        self.quatdim = 4
        self.angveldim = 3
        if J_net is None:
            self.J_net = PSD(self.quatdim, 10, self.angveldim, init_gain=init_gain).to(device)
        else:
            self.J_net = J_net
        if V_net is None:
            self.V_net = MLP(self.quatdim, 10, 1, init_gain=init_gain).to(device)
        else:
            self.V_net = V_net

        self.u_dim = u_dim
        self.h = time_step
        if g_net is None:
            self.g_net = MLP(self.quatdim, 10, self.angveldim).to(device)
        else:
            self.g_net = g_net
        self.device = device
        self.implicit_step = 5

        self.enable_force = enable_force
        self.fixed_J = fixed_J
        self.quat_sym = quat_sym
        self.auto_diff_jacobian = True

        self.nfe = 0

    def forward(self, x):
        with torch.enable_grad():
            self.nfe += 1

            B = x.shape[0]

            qk, wk, uk = torch.split(x, [self.quatdim, self.angveldim, self.u_dim], dim=1) # (B,4), (B,3), (B,u_dim)
            # qk = qk / torch.norm(qk, dim=1, keepdim=True)  # normalize quaternion to avoid error accumulation
            if self.quat_sym:
                J_q_inv = 0.5 * (self.J_net(qk) + self.J_net(-qk))  # (B,3,3) # J(q) = J(-q)
            else:
                J_q_inv = self.J_net(qk) if not self.fixed_J else self.J_net(torch.ones_like(qk))  # (B,3,3)
            if self.quat_sym:
                g_qk = 0.5 * (self.g_net(qk) + self.g_net(-qk)) # f(q) = f(-q)
            else:
                g_qk = self.g_net(qk)

            if self.enable_force:
                fk_minus = self.h * g_qk * uk / 2
                fk_plus = self.h * g_qk * uk / 2
            else:
                fk_minus = torch.zeros(B, self.angveldim, dtype=torch.float64, device=self.device)
                fk_plus = torch.zeros(B, self.angveldim, dtype=torch.float64, device=self.device)

            J_q = torch.inverse(J_q_inv)
            pk = 2 * torch.bmm(J_q, wk.unsqueeze(-1)).squeeze(-1)  # (B,3)

            if self.quat_sym:
                V_qk = 0.5 * (self.V_net(qk) + self.V_net(-qk))  # V(q) = V(-q)
            else:
                V_qk = self.V_net(qk)
            dVk =  torch.autograd.grad(V_qk.sum(), qk, create_graph=True)[0] # (B,4)

            RHS = -self.h / 4 * (pk + fk_minus - self.h / 2 * torch.bmm(H_batch(qk), dVk.unsqueeze(-1)).squeeze(-1))  # (B,3)

            xi = torch.zeros(B, 3, device=self.device, dtype=torch.float64, requires_grad=True)
            for _ in range(self.implicit_step):
                # Compute residual
                q_xi = quat_exp_batch(-xi)
                LHS = torch.bmm(torch.bmm(G_batch(q_xi), J_q), quat_im_batch(q_xi).unsqueeze(-1)).squeeze(-1)
                residual = LHS - RHS  # (B,3)

                # if torch.norm(residual, dim=1).max().item() < 1e-16:
                #     break

                if self.auto_diff_jacobian:
                    # Compute batched Jacobian using autograd
                    jacobian = torch.zeros(B, 3, 3, device=self.device, dtype=torch.float64)
                    for j in range(3):
                        grad = torch.autograd.grad(residual[:, j].sum(), xi, retain_graph=True, create_graph=True)[0]
                        jacobian[:, j, :] = grad

                else:
                    theta = torch.norm(xi, dim=1, keepdim=True).clamp(min=1e-10)  # (B,1)

                    # quaternion
                    q_xi = quat_exp_batch(-xi)
                    
                    sin_theta = torch.sin(theta)
                    cos_theta = torch.cos(theta)

                    I3 = torch.eye(3, device=xi.device, dtype=xi.dtype).unsqueeze(0).repeat(B,1,1)  # (B,3,3)
                    v_skew = skew_batch(xi)  # (B,3,3)
                    outer = xi.unsqueeze(2) @ xi.unsqueeze(1)  # (B,3,3)

                    dIm_dxi = (-cos_theta[:,:,None]*I3
                            + (sin_theta[:,:,None]/theta[:,:,None])*v_skew
                            + ((1 - sin_theta/theta)[:,:,None]/theta[:,:,None]**2)*outer)

                    jacobian = torch.bmm(G_batch(q_xi), J_q) @ dIm_dxi  # (B,3,3)

                # Newton step
                dF_inv = torch.inverse(jacobian)  # (B,3,3)
                xi = xi - torch.bmm(dF_inv, residual.unsqueeze(-1)).squeeze(-1)  # (B,3)

                # print("Residual norm:", torch.norm(residual, dim=1).max().item())

            # Compute next state
            delta_q = quat_exp_batch(xi)  # (B,4)
            qk_next = quat_mul_batch(qk, delta_q)  # (B,4)
            # qk_next = qk_next / torch.norm(qk_next, dim=1, keepdim=True) # normalize quaternion to avoid error accumulation

            # Compute next momentum and angular velocity
            if self.quat_sym:
                V_qk_next = 0.5 * (self.V_net(qk_next) + self.V_net(-qk_next))  # V(q) = V(-q)
            else:
                V_qk_next = self.V_net(qk_next)
            dVk_next = torch.autograd.grad(V_qk_next.sum(), qk_next, create_graph=True)[0]  # (B,4)
            qk_conj = qk * torch.tensor([1, -1, -1, -1], device=self.device, dtype=torch.float64)
            qk_ast_qk_next = quat_mul_batch(qk_conj, qk_next)  # (B,4)
            pk_next = 4 / self.h * torch.bmm(torch.bmm(G_batch(qk_ast_qk_next), J_q), quat_im_batch(qk_ast_qk_next).unsqueeze(-1)).squeeze(-1) \
                        - self.h / 2 * torch.bmm(H_batch(qk_next), dVk_next.unsqueeze(-1)).squeeze(-1) + fk_plus  # (B,3)
            wk_next = 0.5 * torch.bmm(J_q_inv, pk_next.unsqueeze(-1)).squeeze(-1)  # (B,3)

            return torch.cat((qk_next, wk_next, uk), dim=1)
        
    def forward_gt(self, x):
        assert self.enable_force == False, "forward_gt is only for no external force case."
        
        with torch.enable_grad():
            self.nfe += 1

            B = x.shape[0]

            qk, wk, uk = torch.split(x, [self.quatdim, self.angveldim, self.u_dim], dim=1) # (B,4), (B,3), (B,u_dim)
            # qk = qk / torch.norm(qk, dim=1, keepdim=True)  # normalize quaternion to avoid error accumulation
            J_q_inv = 2 * torch.eye(3, device=self.device, dtype=torch.float64).unsqueeze(0).repeat(B,1,1)  # (B,3,3)
            # # g_qk = self.g_net(qk)

            # if self.enable_force:
            #     fk_minus = self.h * g_qk * uk / 2
            #     fk_plus = self.h * g_qk * uk / 2
            # else:
            fk_minus = torch.zeros(B, self.angveldim, dtype=torch.float64, device=self.device)
            fk_plus = torch.zeros(B, self.angveldim, dtype=torch.float64, device=self.device)

            J_q = torch.inverse(J_q_inv)
            pk = 2 * torch.bmm(J_q, wk.unsqueeze(-1)).squeeze(-1)  # (B,3)

            V_qk = qk.sum(dim=1).unsqueeze(-1) # (B,1)
            dVk =  torch.autograd.grad(V_qk.sum(), qk, create_graph=True)[0] # (B,4)

            RHS = -self.h / 4 * (pk + fk_minus - self.h / 2 * torch.bmm(H_batch(qk), dVk.unsqueeze(-1)).squeeze(-1))  # (B,3)

            xi = torch.zeros(B, 3, device=self.device, dtype=torch.float64, requires_grad=True)
            for _ in range(self.implicit_step):
                # Compute residual
                q_xi = quat_exp_batch(-xi)
                LHS = torch.bmm(torch.bmm(G_batch(q_xi), J_q), quat_im_batch(q_xi).unsqueeze(-1)).squeeze(-1)
                residual = LHS - RHS  # (B,3)

                if torch.norm(residual, dim=1).max().item() < 1e-16:
                    break

                # Compute batched Jacobian using autograd
                jacobian = torch.zeros(B, 3, 3, device=self.device, dtype=torch.float64)
                for j in range(3):
                    grad = torch.autograd.grad(residual[:, j].sum(), xi, retain_graph=True, create_graph=True)[0]
                    jacobian[:, j, :] = grad

                # Newton step
                dF_inv = torch.inverse(jacobian)  # (B,3,3)
                xi = xi - torch.bmm(dF_inv, residual.unsqueeze(-1)).squeeze(-1)  # (B,3)

                print("Residual norm:", torch.norm(residual, dim=1).max().item())

            # Compute next state
            delta_q = quat_exp_batch(xi)  # (B,4)
            qk_next = quat_mul_batch(qk, delta_q)  # (B,4)
            # qk_next = qk_next / torch.norm(qk_next, dim=1, keepdim=True) # normalize quaternion to avoid error accumulation

            # Compute next momentum and angular velocity
            V_qk_next = qk_next.sum(dim=1)
            dVk_next = torch.autograd.grad(V_qk_next.sum(), qk_next, create_graph=True)[0]  # (B,4)
            qk_conj = qk * torch.tensor([1, -1, -1, -1], device=self.device, dtype=torch.float64)
            qk_ast_qk_next = quat_mul_batch(qk_conj, qk_next)  # (B,4)
            pk_next = 4 / self.h * torch.bmm(torch.bmm(G_batch(qk_ast_qk_next), J_q), quat_im_batch(qk_ast_qk_next).unsqueeze(-1)).squeeze(-1) \
                        - self.h / 2 * torch.bmm(H_batch(qk_next), dVk_next.unsqueeze(-1)).squeeze(-1) + fk_plus  # (B,3)
            wk_next = 0.5 * torch.bmm(J_q_inv, pk_next.unsqueeze(-1)).squeeze(-1)  # (B,3)

            return torch.cat((qk_next, wk_next, uk), dim=1)
        
    def predict(self, step_num, x, output_rep='quat', verbose=False):
        assert output_rep in ['quat', 'rotmat'], "output_rep must be 'quat' or 'rotmat'"

        if output_rep == 'rotmat':
            # Convert rotation matrix to quaternion
            rotmats = x[:, :9].reshape(-1, 3, 3)  # (B,3,3)
            quats = batch_rotmat_to_quat(rotmats)  # (B,4)
            x = torch.cat((quats, x[:, 9:]), dim=1)  # (B,7)

        xseq = x[None,:,:]
        curx = x

        pbar = tqdm.tqdm(range(step_num)) if verbose else range(step_num)
        for _ in pbar:
            nextx = self.forward(curx)

            # if not self.training:
            #     nextx[:, :self.quatdim] = nextx[:, :self.quatdim] / torch.norm(nextx[:, :self.quatdim], dim=1, keepdim=True)  # normalize quaternion to avoid error accumulation

            # # Ensure the sign of nextx is consistent with curx
            # dot = torch.sum(curx * nextx, dim=1, keepdim=True)
            # nextx = torch.where(dot < 0, -nextx, nextx) 

            curx = nextx
            xseq = torch.cat((xseq, curx[None,:,:]), dim = 0)

            if not self.training:
                torch.cuda.empty_cache()
                gc.collect()
                curx = curx.detach()
                curx.requires_grad_(True)

        if output_rep == 'rotmat':
            quat_seq = xseq[:, :, :self.quatdim].reshape(-1, self.quatdim)
            rotmat_seq = batch_quat_to_rotmat(quat_seq)  # (T*B, 3, 3)
            rotmat_seq = rotmat_seq.reshape(xseq.shape[0], xseq.shape[1], 9)
            return torch.cat((rotmat_seq, xseq[:, :, self.quatdim:]), dim=2)

        return xseq
    

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = S3FVIN(device=device, enable_force=False).to(device)
    x = torch.randn(5, 4+3+1, dtype=torch.float64).to(device)
    x[:, :4] = x[:, :4] / torch.norm(x[:, :4], dim=1, keepdim=True)  # normalize quaternion
    x[:,4:7] = torch.zeros(5,3,dtype=torch.float64).to(device)
    # x[:, 7] = torch.zeros(5,1,dtype=torch.float64).to(device)
    x = x.requires_grad_(True)

    output = model.predict(10, x)

    # Generate some random unit quaternions and 3D vectors to check batch_rotmat_to_quat
    B = 1000
    q_random = torch.randn(B, 4, dtype=torch.float64).to(device)
    q_random = q_random / torch.norm(q_random, dim=1, keepdim=True)
    R_random = batch_quat_to_rotmat(q_random)
    v_random = torch.randn(B, 3, dtype=torch.float64).to(device)

    q_conj_random = q_random * torch.tensor([1, -1, -1, -1], device=device, dtype=torch.float64)
    v_0 = torch.cat([torch.zeros(B,1, dtype=torch.float64).to(device), v_random], dim=1)

    v_rotated = quat_mul_batch(quat_mul_batch(q_random, v_0), q_conj_random)[:,1:]
    v_rotated_R = torch.bmm(R_random, v_random.unsqueeze(-1)).squeeze(-1)
    assert torch.allclose(v_rotated, v_rotated_R, atol=1e-10), "batch_rotmat_to_quat or quat multiplication is incorrect."
    print(v_rotated[:5])
    print(v_rotated_R[:5])

    # Also check batch_quat_to_rotmat
    R_reconstructed = batch_quat_to_rotmat(batch_rotmat_to_quat(R_random))
    assert torch.allclose(R_random, R_reconstructed, atol=1e-10), "batch_rotmat_to_quat or batch_quat_to_rotmat is incorrect."
    print("Rotation matrix conversion check passed.")
    
    # Check the Hamiltonian is conserved
    qk, wk, uk = torch.split(output, [4, 3, 1], dim=2)
    xi_k = 1/2 * wk
    J_q_inv = 2 * torch.eye(3, device=device, dtype=torch.float64).unsqueeze(0).repeat(output.shape[0], output.shape[1], 1, 1)
    J_q = torch.inverse(J_q_inv)
    T, B = xi_k.shape[0], xi_k.shape[1]
    xi_k = xi_k.reshape(T*B, 3)
    J_q = J_q.reshape(T*B, 3, 3)
    kinetic = 2.0 * torch.bmm(xi_k.unsqueeze(1), torch.bmm(J_q, xi_k.unsqueeze(-1))).squeeze(-1).squeeze(-1)
    kinetic = kinetic.reshape(T, B)
    potential = qk.sum(dim=2)
    H = kinetic + potential

    print("Hamiltonian at each time step:", H)

    # print("wk", wk)

    # print("qk", qk)