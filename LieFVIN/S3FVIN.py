import torch
import kornia

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
    Hq = torch.cat([qv.unsqueeze(-1), Gq], dim=2)  # (B,3,4)
    return Hq

def batch_quat_to_rotmat(q):
    """Batched conversion from unit quaternion to rotation matrix"""
    # Kornia expects (x, y, z, w) order
    q_xyzw = torch.zeros_like(q)
    q_xyzw[:, :3] = q[:, 1:]  # x, y, z
    q_xyzw[:, 3] = q[:, 0]    # w

    R = kornia.geometry.conversions.quaternion_to_rotation_matrix(q_xyzw)  # (B, 3, 3)
    return R

def batch_rotmat_to_quat(R):
    """Batched conversion from rotation matrix to unit quaternion (B,3,3)->(B,4)."""
    # kornia expects shape (B, 3, 3) and returns (B, 4) in (x, y, z, w) order
    q_xyzw = kornia.geometry.conversions.rotation_matrix_to_quaternion(R)  # (B,4), (x,y,z,w)
    
    # reorder to (w,x,y,z)
    q_wxyz = torch.zeros_like(q_xyzw)
    q_wxyz[:, 0] = q_xyzw[:, 3]  # w
    q_wxyz[:, 1:] = q_xyzw[:, :3]  # x,y,z
    
    return q_wxyz


class S3FVIN(torch.nn.Module):
    def __init__(self, J_net=None, V_net=None, g_net=None, device=None, u_dim=1, time_step=0.01, init_gain=1):
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
        self.implicit_step = 4

        self.nfe = 0

    def forward(self, x):
        enable_force = True
        
        with torch.enable_grad():
            self.nfe += 1

            B = x.shape[0]

            qk, wk, uk = torch.split(x, [self.quatdim, self.angveldim, self.u_dim], dim=1) # (B,4), (B,3), (B,u_dim)
            J_q_inv = self.J_net(qk)
            g_qk = self.g_net(qk)

            if enable_force:
                fk_minus = self.h * g_qk * uk / 2
                fk_plus = self.h * g_qk * uk / 2
            else:
                fk_minus = torch.zeros(B, self.angveldim, dtype=torch.float64, device=self.device)
                fk_plus = torch.zeros(B, self.angveldim, dtype=torch.float64, device=self.device)

            J_q = torch.inverse(J_q_inv)
            pk = 2 * torch.bmm(J_q, wk.unsqueeze(-1)).squeeze(-1)  # (B,3)

            V_qk = self.V_net(qk)
            dVk =  torch.autograd.grad(V_qk.sum(), qk, create_graph=True)[0] # (B,4)

            RHS = -self.h / 4 * (pk + fk_minus - self.h / 2 * torch.bmm(H_batch(qk), dVk.unsqueeze(-1)).squeeze(-1))  # (B,3)

            xi = torch.zeros(B, 3, device=self.device, dtype=torch.float64, requires_grad=True)
            for _ in range(self.implicit_step):
                # Compute residual
                q_xi = quat_exp_batch(-xi)
                LHS = torch.bmm(torch.bmm(G_batch(q_xi), J_q), quat_im_batch(q_xi).unsqueeze(-1)).squeeze(-1)
                residual = LHS - RHS  # (B,3)

                # Compute batched Jacobian using autograd
                jacobian = torch.zeros(B, 3, 3, device=self.device, dtype=torch.float64)
                for j in range(3):
                    grad = torch.autograd.grad(residual[:, j].sum(), xi, retain_graph=True, create_graph=True)[0]
                    jacobian[:, j, :] = grad

                # Newton step
                dF_inv = torch.inverse(jacobian)  # (B,3,3)
                xi = xi - torch.bmm(dF_inv, residual.unsqueeze(-1)).squeeze(-1)  # (B,3)

                # print("Residual norm:", torch.norm(residual, dim=1).max().item())

            # Compute next state and momentum
            delta_q = quat_exp_batch(xi)  # (B,4)
            qk_next = quat_mul_batch(qk, delta_q)  # (B,4)
            V_qk_next = self.V_net(qk_next)
            dVk_next = torch.autograd.grad(V_qk_next.sum(), qk_next, create_graph=True)[0]  # (B,4)
            qk_conj = qk * torch.tensor([1, -1, -1, -1], device=self.device, dtype=torch.float64)
            qk_ast_qk_next = quat_mul_batch(qk_conj, qk_next)  # (B,4)
            pk_next = 4 / self.h * torch.bmm(torch.bmm(G_batch(qk_ast_qk_next), J_q), quat_im_batch(qk_ast_qk_next).unsqueeze(-1)).squeeze(-1) \
                        - self.h / 2 * torch.bmm(H_batch(qk_next), dVk_next.unsqueeze(-1)).squeeze(-1) + fk_plus  # (B,3)
            wk_next = 0.5 * torch.bmm(J_q_inv, pk_next.unsqueeze(-1)).squeeze(-1)  # (B,3)

            return torch.cat((qk_next, wk_next, uk), dim=1)

    def predict(self, step_num, x, output_rep='quat'):
        assert output_rep in ['quat', 'rotmat'], "output_rep must be 'quat' or 'rotmat'"

        if output_rep == 'rotmat':
            # Convert rotation matrix to quaternion
            rotmats = x[:, :9].reshape(-1, 3, 3)  # (B,3,3)
            quats = batch_rotmat_to_quat(rotmats)  # (B,4)
            x = torch.cat((quats, x[:, 9:]), dim=1)  # (B,7)

        xseq = x[None,:,:]
        curx = x

        for _ in range(step_num):
            nextx = self.forward(curx)

            # Ensure the sign of nextx is consistent with curx
            dot = torch.sum(curx * nextx, dim=1, keepdim=True)
            nextx = torch.where(dot < 0, -nextx, nextx) 

            curx = nextx
            xseq = torch.cat((xseq, curx[None,:,:]), dim = 0)

        if output_rep == 'rotmat':
            # Convert quaternion to rotation matrix and flatten
            quat_seq = xseq[:, :, :self.quatdim].reshape(-1, self.quatdim) # (step_num+1)*B, 4
            rotmat_seq = batch_quat_to_rotmat(quat_seq).reshape(-1, 9) # (step_num+1)*B, 9
            return torch.cat((rotmat_seq.reshape(step_num+1, -1, 9), xseq[:, :, self.quatdim:]), dim=2)

        return xseq
    

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = S3FVIN(device=device).to(device)
    x = torch.randn(10, 4+3+1, dtype=torch.float64).to(device).requires_grad_(True)
    y = model(x)
    
    # Generate random unit quaternions and check conversion
    q = torch.randn(1000,4, dtype=torch.float64).to(device)
    q = q / torch.norm(q, dim=1, keepdim=True)
    R = batch_quat_to_rotmat(q)
    q_rec = batch_rotmat_to_quat(R)

    # Ensure the quaternions are equivalent up to sign
    dot = torch.sum(q * q_rec, dim=1, keepdim=True)
    q_rec = torch.where(dot < 0, -q_rec, q_rec)

    print("Max quaternion conversion error:", torch.norm(q - q_rec, dim=1).max().item())