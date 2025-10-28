import os
import inspect
n_cores = 4
os.environ["XLA_FLAGS"] = '--xla_force_host_platform_device_count={}'.format(n_cores)

from jax import config
config.update("jax_enable_x64", True)

import jax
import jax.numpy as jnp
import numpy as np
import pickle
import meshio
from jax import jit, vmap, grad, jacrev, lax
from jax.lax import cond, scan
import jax.random as random
key = random.PRNGKey(2022)
import jax.example_libraries.optimizers as optimizers
from jax.flatten_util import ravel_pytree
from jax.sharding import PositionalSharding
from functools import partial

import matplotlib.pyplot as plt
import pandas as pd

from utils import train, coords_2_strain_nn

from utils import train_colloc_parallel as train_colloc,init_params_nn, ff_nn, divergence, bd_forces, a1, a2, lr
from utils_hyperelasticity import NODE, init_layers, NODE_model_aniso, init_params_aniso, GOH_model, eval_Cauchy, eval_Cauchy_vmap, eval_P, eval_P_vmap
from utils_hyperelasticity import ThreeDElasticity
from fem import plotmesh, fe_solver_2D, vahid_anisohyper_inv, apply_bc_biax



from PIL import Image, ImageFilter
from typing import Iterable, Optional

const_params = jnp.array([0.5, 0.10, 0.20, 0.10, 0.0])

def coords_2_params_gt(xy):
    # xy ignored,  uniform material
    return const_params
coords_2_params_gt_v = jax.vmap(coords_2_params_gt, in_axes=(0,))
with open('hole_paper_pre.npy', 'rb') as f:
    coord_2_strain_params, node_params, Lambda_params, mesh, elem_X, Force_x, Force_y, strains, \
                 bd_X, lmb_hist, P_hist_goh, F_hist_goh, t_hist = pickle.load(f)
_, unravel = ravel_pytree(node_params)
NODE_w_unravel = lambda params: NODE_model_aniso(unravel(params))
coords_2_strain_nn_ff = lambda x, params: coords_2_strain_nn(x, params)
ff_nn_ff = lambda x, params: ff_nn(x, params)


#%%
Lx, Ly = 1.0, 1.0
atol = 1e-8

# ensure 2D coords
pts = mesh.points[:, :2]
conn = mesh.cells  # (Ne, 4)

# local edge pairs (0-1, 1-2, 2-3, 3-0)
edge_pairs = np.array([[0,1],[1,2],[2,3],[3,0]], dtype=int)

# outputs
rgt_bd_elems, top_bd_elems, lft_bd_elems, bot_bd_elems = [], [], [], []
rgt_bd_lengths, top_bd_lengths, lft_bd_lengths, bot_bd_lengths = [], [], [], []

for ei, e in enumerate(conn):
    xy_e = pts[e]                               
    p0 = xy_e[edge_pairs[:,0]]                 
    p1 = xy_e[edge_pairs[:,1]]                  
    mid = 0.5 * (p0 + p1)                       

    dx = np.abs(p1[:,0] - p0[:,0])              
    dy = np.abs(p1[:,1] - p0[:,1])              

    
    ks = np.where(np.isclose(mid[:,0], Lx, atol=atol))[0]
    for k in ks:
        rgt_bd_elems.append(ei)
        rgt_bd_lengths.append(dy[k])

    
    ks = np.where(np.isclose(mid[:,0], 0.0, atol=atol))[0]
    for k in ks:
        lft_bd_elems.append(ei)
        lft_bd_lengths.append(dy[k])

    
    ks = np.where(np.isclose(mid[:,1], Ly, atol=atol))[0]
    for k in ks:
        top_bd_elems.append(ei)
        top_bd_lengths.append(dx[k])

    
    ks = np.where(np.isclose(mid[:,1], 0.0, atol=atol))[0]
    for k in ks:
        bot_bd_elems.append(ei)
        bot_bd_lengths.append(dx[k])


rgt_bd_elems   = np.asarray(rgt_bd_elems,   dtype=int)
top_bd_elems   = np.asarray(top_bd_elems,   dtype=int)
lft_bd_elems   = np.asarray(lft_bd_elems,   dtype=int)
bot_bd_elems   = np.asarray(bot_bd_elems,   dtype=int)

rgt_bd_lengths = np.asarray(rgt_bd_lengths, dtype=float)
top_bd_lengths = np.asarray(top_bd_lengths, dtype=float)
lft_bd_lengths = np.asarray(lft_bd_lengths, dtype=float)
bot_bd_lengths = np.asarray(bot_bd_lengths, dtype=float)
#%%


rgt_bd_frc = np.sum(P_hist_goh[:,rgt_bd_elems,0,0]*rgt_bd_lengths, axis=1)
lft_bd_frc = np.sum(P_hist_goh[:,lft_bd_elems,0,0]*lft_bd_lengths , axis=1)
top_bd_frc = np.sum(P_hist_goh[:,top_bd_elems,1,1]*top_bd_lengths, axis=1)
bot_bd_frc = np.sum(P_hist_goh[:,bot_bd_elems,1,1]*bot_bd_lengths, axis=1)
Force_x = jnp.array(0.5*(rgt_bd_frc + lft_bd_frc))
Force_y = jnp.array(0.5*(top_bd_frc + bot_bd_frc))

# get the points of the elements
rgt_bd_X = elem_X[rgt_bd_elems]
top_bd_X = elem_X[top_bd_elems]
lft_bd_X = elem_X[lft_bd_elems]
bot_bd_X = elem_X[bot_bd_elems]
bd_X = rgt_bd_X, top_bd_X, lft_bd_X, bot_bd_X
#%%
with open('hole_paper_second_pretraining.pickle', 'rb') as f:
    Lambda_params,val_loss, metrics=pickle.load(f)
E_fun = vmap(lambda F: 0.5*(F.T@F - jnp.eye(2)))
#%% functions that generate meshes
def q4_gauss_2x2():
    g = 1.0 / np.sqrt(3.0)
    gp = np.array([[-g, -g],
                   [ g, -g],
                   [ g,  g],
                   [-g,  g]], dtype=float)  # (4,2)
    w = np.ones(4, dtype=float)
    return gp, w

def q4_shape_and_parent_grads(xi, eta):
    
    N = 0.25*np.array([
        (1 - xi)*(1 - eta),
        (1 + xi)*(1 - eta),
        (1 + xi)*(1 + eta),
        (1 - xi)*(1 + eta),
    ], dtype=float)

    dN_dxi = 0.25*np.array([
        -(1 - eta),
         (1 - eta),
         (1 + eta),
        -(1 + eta),
    ], dtype=float)

    dN_deta = 0.25*np.array([
        -(1 - xi),
        -(1 + xi),
         (1 + xi),
         (1 - xi),
    ], dtype=float)

    return N, dN_dxi, dN_deta

def build_node_to_elem(conn, n_nodes):
    node_to_elem = [[] for _ in range(n_nodes)]
    for e, c in enumerate(conn):
        for n in c:
            node_to_elem[n].append(e)
    return node_to_elem

def compute_q4_gauss_data(nodes, conn):
    Ne = conn.shape[0]
    gp_parent, _ = q4_gauss_2x2()

    Xgp   = np.zeros((Ne,4,2), dtype=float)
    J     = np.zeros((Ne,4,2,2), dtype=float)
    detJ  = np.zeros((Ne,4), dtype=float)
    dN_dx = np.zeros((Ne,4,4,2), dtype=float)
    Nvals = np.zeros((Ne,4,4), dtype=float)

    for e in range(Ne):
        x_e = nodes[conn[e], :]  # (4,2), order [bl, br, tr, tl]
        for gpi, (xi, eta) in enumerate(gp_parent):
            N, dN_dxi, dN_deta = q4_shape_and_parent_grads(xi, eta)
            Nvals[e, gpi, :] = N

            # Map Gauss point to physical space
            Xgp[e, gpi, :] = N @ x_e

            
            dx_dxi  = dN_dxi  @ x_e[:, 0]
            dx_deta = dN_deta @ x_e[:, 0]
            dy_dxi  = dN_dxi  @ x_e[:, 1]
            dy_deta = dN_deta @ x_e[:, 1]
            Je = np.array([[dx_dxi, dx_deta],
                           [dy_dxi, dy_deta]], dtype=float)
            J[e, gpi]    = Je
            detJ[e, gpi] = np.linalg.det(Je)

            try:
                invJT   = np.linalg.inv(Je).T
            except np.linalg.LinAlgError:
                invJT = np.full((2,2), np.nan)
            dN_par  = np.vstack((dN_dxi, dN_deta))   
            dN_phys = invJT @ dN_par                 
            dN_dx[e, gpi, :, :] = dN_phys.T           

    return Xgp, J, detJ, dN_dx, Nvals, gp_parent


def compute_node_gauss_data(nodes, conn):
    """
    Build a dictionary with per-node Gauss data:
        node_data[n] = {
            'elem_ids'   : (Me,)  element IDs connected to node n
            'local_ids'  : (Me,)  local index of node n inside each element
            'Xgp_node'   : (Me,4,2) physical Gauss point coordinates for elems
            'J_node'     : (Me,4,2,2) Jacobians at gp
            'detJ_node'  : (Me,4) determinant of J at gp
            'dNdx_node'  : (Me,4,2) gradient of shape function for node n
            'N_node'     : (Me,4) shape function value for node n
        }
    """
    Xgp, J, detJ, dN_dx, Nvals, gp_parent = compute_q4_gauss_data(nodes, conn)

    n_nodes = nodes.shape[0]
    node2elem = build_node_to_elem(conn, n_nodes)

    node_data = {}  # <-- dictionary keyed by node index

    for n in range(n_nodes):
        elems = node2elem[n]
        if len(elems) == 0:
            node_data[n] = {
                'elem_ids': np.array([], dtype=int),
                'local_ids': np.array([], dtype=int),
                'Xgp_node': np.zeros((0,4,2)),
                'J_node': np.zeros((0,4,2,2)),
                'detJ_node': np.zeros((0,4)),
                'dNdx_node': np.zeros((0,4,2)),
                'N_node': np.zeros((0,4)),
            }
            continue

        # local node index within each element
        local_ids = np.array([int(np.where(conn[e] == n)[0][0]) for e in elems], dtype=int)

        node_data[n] = {
            'elem_ids'  : np.array(elems, dtype=int),
            'local_ids' : local_ids,
            'Xgp_node'  : Xgp[elems, :, :],
            'J_node'    : J[elems, :, :, :],
            'detJ_node' : detJ[elems, :],
            'dNdx_node' : np.stack([dN_dx[e, :, m, :] for e, m in zip(elems, local_ids)], axis=0),
            'N_node'    : np.stack([Nvals[e, :, m]   for e, m in zip(elems, local_ids)], axis=0),
        }

    return node_data
def pack_node_data(node_data):
    """
    Pack ragged per-node Gauss data into uniform padded arrays.

    Returns:
      Xgp_all     : (Nn, Me_max, 4, 2)
      detJ_all    : (Nn, Me_max, 4)
      dNdx_all    : (Nn, Me_max, 4, 2)
      valid_mask  : (Nn, Me_max, 4) -> 1.0 where valid, 0.0 where padded
    """
    Nn = len(node_data)
    Me_list = [node_data[n]['Xgp_node'].shape[0] for n in range(Nn)]
    Me_max = max(Me_list) if Nn > 0 else 0

    Xgp_all    = np.zeros((Nn, Me_max, 4, 2), dtype=float)
    detJ_all   = np.zeros((Nn, Me_max, 4), dtype=float)
    dNdx_all   = np.zeros((Nn, Me_max, 4, 2), dtype=float)
    valid_mask = np.zeros((Nn, Me_max, 4), dtype=float)

    for n in range(Nn):
        Me = Me_list[n]
        if Me == 0: 
            continue
        Xgp_all[n, :Me]    = node_data[n]['Xgp_node']     # (Me,4,2)
        detJ_all[n, :Me]   = node_data[n]['detJ_node']    # (Me,4)
        dNdx_all[n, :Me]   = node_data[n]['dNdx_node']    # (Me,4,2)
        valid_mask[n, :Me] = 1.0                          # mark valid gauss points

    # Convert to JAX
    return (jnp.array(Xgp_all),
            jnp.array(detJ_all),
            jnp.array(dNdx_all),
            jnp.array(valid_mask))
def element_stress_from_node_data(P_avg_2x2, node_data, conn):
    """
    P_avg_2x2 : (Nn, Me<=4, 2, 2)  element-averaged stress per (node, incident-element-slot)
    node_data : dict from compute_node_gauss_data(...)  (has 'elem_ids' per node)
    conn      : (Ne, 4) connectivity to get Ne

    Returns:
        P_elem : (Ne, 2, 2) element-wise averaged stress
        counts : (Ne,) how many node-contributions were averaged (corner=1, edge=2, interior=4)
    """
    Ne = conn.shape[0]
    P_sum   = np.zeros((Ne, 2, 2), dtype=float)
    counts  = np.zeros(Ne, dtype=int)

    # loop nodes; only use the first len(elem_ids) slots (ignore padding)
    for n, nd in node_data.items():
        elems = nd['elem_ids']           # shape (k,), k∈{1,2,3,4}
        k = int(len(elems))
        if k == 0:
            continue
        Pe = np.asarray(P_avg_2x2[n, :k, :, :])   # (k, 2, 2)
        for e_idx, P in zip(elems, Pe):
            P_sum[e_idx] += P
            counts[e_idx] += 1

    # avoid division by zero (shouldn't happen if mesh is consistent)
    assert np.all(counts > 0), "Some elements received no node contributions."
    P_elem = P_sum / counts[:, None, None]
    return P_elem, counts
def eval_divP_term_batch(node_ids, time_ids, t_hist,
                         Xgp_all, detJ_all, dNdx_all, valid_mask,
                         P_func):
    """
    Vectorized evaluation over a batch of (node_id, time_id).
    Inputs:
      node_ids   : (B,) int
      time_ids   : (B,) int
      t_hist     : (Nt,)
      Xgp_all    : (Nn, Me_max, 4, 2)
      detJ_all   : (Nn, Me_max, 4)
      dNdx_all   : (Nn, Me_max, 4, 2)
      valid_mask : (Nn, Me_max, 4)  in {0,1}
      P_func     : vectorized P(x,y,t) -> (...,2,2) over broadcasted x,y,t

    Returns:
      I_batch : (B, 2) with components [I1, I2] per sampled address.
    """
    # 1) Gather per-batch node Gauss data
    XgpB    = Xgp_all[node_ids]        
    detJB   = detJ_all[node_ids]      
    dNdxB   = dNdx_all[node_ids]       
    maskB   = valid_mask[node_ids]    

    
    tB = t_hist[time_ids]             
    tB = tB[:, None, None]            

    
    x = XgpB[..., 0]                  
    y = XgpB[..., 1]                  
    P = P_func(x, y, tB)              

    # 4) Contract integrand: for i=0 and i=1
    Nx = dNdxB[..., 0]                
    Ny = dNdxB[..., 1]                

    # P[..., i,1]*Nx + P[..., i,2]*Ny
    integrand_i0 = P[..., 0, 0] * Nx + P[..., 0, 1] * Ny
    integrand_i1 = P[..., 1, 0] * Nx + P[..., 1, 1] * Ny

   
    w = 1.0  # 2x2 Gauss weights are 1 for each gp
    scale = detJB * maskB * w

    I0 = jnp.sum(integrand_i0 * scale, axis=(1, 2))  # (B,)
    I1 = jnp.sum(integrand_i1 * scale, axis=(1, 2))  # (B,)

    return jnp.stack([I0, I1], axis=1) 
def get_P(X, Y, t, Lambda_params, coord_2_strain_params, model):
    F_xx, F_xy, F_yx, F_yy = coords_2_strain_nn_ff(jnp.array([X,Y,t])[None,:], coord_2_strain_params).flatten()
    # get NODE individual-specific params, phi, from the Lambda NN
    Lambda_inp = jnp.array([X,Y]).reshape([-1,2])
    phi = ff_nn_ff(Lambda_inp, Lambda_params).flatten()
    # Make predictions with this NODE
    ugrad = jnp.array([[F_xx-1.0, F_xy],[F_yx, F_yy-1.0]])
    P = ThreeDElasticity(model).ugrad_2_P(ugrad, phi, 2)
    return P[0,0], P[1,0], P[0,1], P[1,1]
get_P_vmap = vmap(get_P, in_axes=(0,0,0,None,None,None), out_axes=(0,0,0,0))
def get_side_nodes(grid):
    left   = grid[:, 0]
    right  = grid[:, -1]
    bottom = grid[0, :]
    top    = grid[-1, :]
    return left, right, top, bottom

def side_nodes(grid, side):
   
    side = side.lower()
    if side == "left":   return grid[:, 0]
    if side == "right":  return grid[:, -1]
    if side == "bottom": return grid[0, :]
    if side in ("top", "up"): return grid[-1, :]
    raise ValueError("side must be one of: 'left','right','bottom','top' (or 'up').")
def assemble_constant_from_reactions(nodes,Lx, Ly, reactions, thickness=1.0):
    """
    reactions: dict like {"left": [Fx,Fy], "right":[Fx,Fy], "bottom":[Fx,Fy], "top":[Fx,Fy]}
               Fx,Fy are TOTAL forces on that side (not per-length).
    Constant traction on each side:  t = F_total / side_length
    Edge load for 2-node edge with constant t:  [t*L/2, t*L/2]  (× thickness)
    """
    N = nodes.shape[0]
    f1 = np.zeros(N); f2 = np.zeros(N)

    def side_ids(side):
        s = side.lower()
        if s == "left":   return np.array(np.where(np.abs(nodes[:,0]-0)<10e-4)[0])[0]
        if s == "right":  return np.array(np.where(np.abs(nodes[:,0]-Lx)<10e-4)[0])[0]
        if s == "bottom": return np.array(np.where(np.abs(nodes[:,1]-0)<10e-4)[0])[0]
        if s in ("top","up"): return np.array(np.where(np.abs(nodes[:,1]-Ly)<10e-4)[0])[0]
        raise ValueError("side must be left/right/bottom/top")

    def side_length(side):
        s = side.lower()
        if s in ("left","right"):  return Ly
        if s in ("bottom","top","up"): return Lx
        raise ValueError

    for side, F in reactions.items():
        Fx, Fy = np.asarray(F, float)
        Ls = side_length(side)
        tx_const = Fx / Ls
        ty_const = Fy / Ls

        ids = edges[side]['nodes']
        for a, b in zip(ids[:-1], ids[1:]):
            xa, ya = nodes[a]; xb, yb = nodes[b]
            L = np.hypot(xb - xa, yb - ya)
            w = thickness * L / 2.0  # constant traction ⇒ [t*L/2, t*L/2]

            f1[a] += tx_const * w; f1[b] += tx_const * w
            f2[a] += ty_const * w; f2[b] += ty_const * w

    return f1, f2
def side_length(Lx, Ly, side):
    """Geometric length of a boundary side in the rectangle."""
    s = side.lower()
    if s in ("left", "right"):
        return Ly
    if s in ("bottom", "top", "up"):
        return Lx
    raise ValueError("side must be one of: 'left','right','bottom','top' (or 'up').")

def distribute_reactions_to_tractions(grid, Lx, Ly, reactions_dict):
    N = grid.size
    tx = np.zeros(N)
    ty = np.zeros(N)
    info = {}

    for side, F in reactions_dict.items():
        F = np.asarray(F, dtype=float)   # [Fx, Fy]
        nodes_on_side = side_nodes(grid, side)
        Ns = nodes_on_side.size
        Ls = side_length(Lx, Ly, side)
        t_node = F / (Ls)

        tx[nodes_on_side] += t_node[0]
        ty[nodes_on_side] += t_node[1]

        info[side.lower()] = {
            "count": int(Ns),
            "length": float(Ls),
            "t_per_side": t_node.copy(),
            "node_ids": nodes_on_side.copy(),
        }

    return tx, ty, info

#%%  General Loss functions
@partial(jit, static_argnums=(0,2,3,))
def step_colloc_weak(loss, i, get_params, opt_update, opt_state,
                     spatial_temporal_address, XYt_colloc, dN_dx,maskB):
    params = get_params(opt_state)
    g = grad(loss)(params, XYt_colloc, dN_dx, spatial_temporal_address,maskB)
    return opt_update(i, g, opt_state)



def train_colloc_parallel_weak(loss, inp, gp_data_nodewise,
                               get_params, opt_update, opt_state,
                               key, sharding=None, fname=None,
                               nIter=2000, print_freq=1000, metric_fns=None, batch_size=8192):
    
    """
    Vectorized training with (node_id, time_id) sampling.
    gp_data_nodewise = [Xgp_all, detJ_all, dNdx_all, valid_mask]
    inp = [node_X, t_hist]
    """
    Xgp_all, detJ_all, dNdx_all, valid_mask = gp_data_nodewise
    node_X, t_hist = inp
    Nn = node_X.shape[0]
    Nt = t_hist.shape[0]

    val_loss, metrics = [], []

    for it in range(nIter):
        key, sk1 = random.split(key, 2)
        total = Nn * Nt
        row_idx  = random.randint(sk1, (batch_size,), 0, total)     # (B,)
        node_ids = (row_idx % Nn).astype(jnp.int32)                 # (B,)
        time_ids = (row_idx // Nn).astype(jnp.int32)                # (B,)

        # Gather Gauss data for these nodes
        XgpB  = jnp.take(Xgp_all,  node_ids, axis=0)   # (B, Me, 4, 2)
        #params_B =jnp.take(params_np,  node_ids, axis=0)
        detJB = jnp.take(detJ_all, node_ids, axis=0)   # (B, Me, 4)
        dNdxB = jnp.take(dNdx_all, node_ids, axis=0)   # (B, Me, 4, 2)
        maskB = jnp.take(valid_mask, node_ids, axis=0) # (B, Me, 4)

        # Broadcast time to Gauss grid
        tB      = t_hist[time_ids]                                        # (B,)
        tB_full = jnp.broadcast_to(tB[:, None, None], XgpB.shape[:3])     # (B,Me,4)

        # Build Gauss-grid XYt_colloc = [x,y,t]
        x = XgpB[..., 0]; y = XgpB[..., 1]
        XYt_colloc = jnp.stack([x, y, tB_full], axis=-1)                  # (B,Me,4,3)
                # Pre-weight gradients with detJ & mask → plain sum in loss
        dNdxB_w = dNdxB * detJB[..., None] * maskB[..., None]             # (B,Me,4,2)

        addr = jnp.stack([node_ids, time_ids], axis=1)                    # (B,2)

        opt_state = step_colloc_weak(loss, it, get_params, opt_update, opt_state,
                                     addr, XYt_colloc, dNdxB_w,maskB)

        #%%
        if (it % print_freq) == 0:
            params = get_params(opt_state)
            # Validate at last time for all nodes
            node_ids_v = jnp.arange(Nn, dtype=jnp.int32)
            time_ids_v = jnp.full((Nn,), Nt-1, dtype=jnp.int32)

            XgpV  = jnp.take(Xgp_all,  node_ids_v, axis=0)
            detJV = jnp.take(detJ_all, node_ids_v, axis=0)
            dNdxV = jnp.take(dNdx_all, node_ids_v, axis=0)
            maskV = jnp.take(valid_mask, node_ids_v, axis=0)

            tV      = jnp.full((Nn,), t_hist[-1])
            tV_full = jnp.broadcast_to(tV[:, None, None], XgpV.shape[:3])
            XYt_val = jnp.stack([XgpV[..., 0], XgpV[..., 1], tV_full], axis=-1)
            dNdxV_w = dNdxV * detJV[..., None] * maskV[..., None]

            addr_v = jnp.stack([node_ids_v, time_ids_v], axis=1)
            vl = loss(params, XYt_val, dNdxV_w, addr_v,maskV )
            val_loss.append(vl)
            print(f"it {it+1}, val loss = {float(vl):.6e}")

            if metric_fns is not None:
                m = []
                for metric_fn in metric_fns:
                    m.append(metric_fn(params, jnp.hstack([node_X, t_hist[-1]*jnp.ones((Nn,1))])))
                metrics.append(m)

    return get_params(opt_state), val_loss, metrics
#%%
def stress_masked_one(xyt, is_valid, Lambda_params, coord_2_strain_params, model):
    # All-in-one: cond + get_P + packing
    def _compute(xx):
        X, Y, t = xx
        # strains/gradients
        F_xx, F_xy, F_yx, F_yy = coords_2_strain_nn_ff(xx[None, :], coord_2_strain_params).flatten()
        # material/phi
        phi = ff_nn_ff(jnp.array([X, Y]).reshape(1, 2), Lambda_params).flatten()
        
        ugrad = jnp.array([[F_xx - 1.0, F_xy],
                           [F_yx,       F_yy - 1.0]])
        P = ThreeDElasticity(model).ugrad_2_P(ugrad, phi, 2)
        return jnp.array([P[0, 0], P[1, 0], P[0, 1], P[1, 1]], dtype=P.dtype)
    # # Use xyt to define dtype for the "false" branch
    return lax.cond(is_valid,
                    _compute,
                    lambda xx: jnp.zeros((4,), dtype=xx.dtype),
                    xyt)
stress_masked_one_jit = jit(stress_masked_one, static_argnums=(4,))


v_stress = vmap(
    vmap(
        vmap(
            stress_masked_one_jit,
            in_axes=(0, 0, None, None, None),   
            out_axes=0
        ),
        in_axes=(0, 0, None, None, None),
        out_axes=0
    ),
    in_axes=(0, 0, None, None, None),
    out_axes=0
)
@jit
def loss_weak_form(Lambda_params, XYt_colloc, dN_dx, 
                   spatial_temporal_address,maskB
                   , alpha=1, eps=1e-9):
    node_ids = spatial_temporal_address[:, 0].astype(jnp.int32)
    time_ids = spatial_temporal_address[:, 1].astype(jnp.int32)
    shp = XYt_colloc[..., 0].shape
    stresses=v_stress(XYt_colloc, maskB, Lambda_params, coord_2_strain_params, NODE_w_unravel)
    # P at Gauss points via your get_P_vmap (returns P00,P10,P01,P11)
    P00 = stresses[..., 0]  # shape (B, Me, 4)
    P10 = stresses[..., 1]  # shape (B, Me, 4)
    P01 = stresses[..., 2]  # shape (B, Me, 4)
    P11 = stresses[..., 3]  # shape (B, Me, 4)
    P00 = P00.reshape(shp); P10 = P10.reshape(shp)
    P01 = P01.reshape(shp); P11 = P11.reshape(shp)

    N_x = dN_dx[..., 0]  # (B,Me,4) detJ*mask-weighted
    N_y = dN_dx[..., 1]

    I0 = jnp.sum(P00 * N_x + P01 * N_y, axis=(1, 2))  # (B,)
    I1 = jnp.sum(P10 * N_x + P11 * N_y, axis=(1, 2))  # (B,)

    tx = tx_hist[time_ids, node_ids]  # (B,)
    ty = ty_hist[time_ids, node_ids]  # (B,)

    r0 = I0 - tx
    r1 = I1 - ty
    w0 = jnp.where(jnp.abs(tx) <eps, alpha, 1.0)
    w1 = jnp.where(jnp.abs(ty) < eps, alpha, 1.0)
    return jnp.sum(w0 * r0**2 + w1 * r1**2)
#%%

def ogrid_plate_with_hole(Ntheta=128, Nr=40, Lx=1.0, Ly=1.0, cx=0.5, cy=0.5, r=0.20):
    """
    Build a boundary-fitted QUAD4 O-grid for a square with a circular hole.
    Returns:
      points2d: (Npts, 2) float array
      quads   : (Ncells, 4) int array (CCW)
    """
    thetas = np.linspace(0.0, 2.0*np.pi, Ntheta, endpoint=False)

    
    def hit_square(theta):
        dx, dy = np.cos(theta), np.sin(theta)
        eps = 1e-14
        hits = []
        if abs(dx) > eps:
            t = (0.0 - cx) / dx; y = cy + t*dy
            if t > 0 and -eps <= y <= Ly + eps: hits.append((t, 0.0, y))
            t = (Lx - cx) / dx; y = cy + t*dy
            if t > 0 and -eps <= y <= Ly + eps: hits.append((t, Lx,  y))
        if abs(dy) > eps:
            t = (0.0 - cy) / dy; x = cx + t*dx
            if t > 0 and -eps <= x <= Lx + eps: hits.append((t, x, 0.0))
            t = (Ly - cy) / dy; x = cx + t*dx
            if t > 0 and -eps <= x <= Lx + eps: hits.append((t, x,  Ly))
        tmin, bx, by = sorted(hits, key=lambda a: a[0])[0]
        return bx, by

    outer_pts = np.array([hit_square(th) for th in thetas])               # (Nθ,2)
    inner_pts = np.column_stack([cx + r*np.cos(thetas), cy + r*np.sin(thetas)])  # (Nθ,2)

    
    points2d = np.zeros(((Nr+1)*Ntheta, 2), dtype=float)
    for j in range(Ntheta):
        xi, yi = inner_pts[j]
        bx, by = outer_pts[j]
        for i in range(Nr+1):
            t = i / Nr
            points2d[i*Ntheta + j, 0] = (1.0 - t)*xi + t*bx
            points2d[i*Ntheta + j, 1] = (1.0 - t)*yi + t*by

    
    quads = []
    for i in range(Nr):
        for j in range(Ntheta):
            jn = (j + 1) % Ntheta
            n00 = i    * Ntheta + j
            n01 = i    * Ntheta + jn
            n11 = (i+1)* Ntheta + jn
            n10 = (i+1)* Ntheta + j
            quads.append([n00, n01, n11, n10])
    quads = np.asarray(quads, dtype=int)
    quads = quads[:, [0, 3, 2, 1]] 
    return points2d, quads
def order_outer_boundary(points2d, Lx=1.0, Ly=1.0, atol=1e-8):
    """
    Return ordered node indices and segment pairs for each outer edge.
    Ordering:
      bottom: x increasing
      right : y increasing
      top   : x increasing   <-- your request
      left  : y increasing   <-- your request

    Note: this per-edge ordering is NOT a single CCW loop.
          Use the 'ccw_loop' section below if you need a global CCW loop.
    """
    x = points2d[:, 0]; y = points2d[:, 1]

    bottom_ids = np.where(np.isclose(y, 0.0, atol=atol))[0]
    right_ids  = np.where(np.isclose(x, Lx,  atol=atol))[0]
    top_ids    = np.where(np.isclose(y, Ly,  atol=atol))[0]
    left_ids   = np.where(np.isclose(x, 0.0, atol=atol))[0]

    bottom_sorted = bottom_ids[np.argsort(x[bottom_ids])]   # x ↑
    right_sorted  = right_ids[np.argsort(y[right_ids])]     # y ↑
    top_sorted    = top_ids[np.argsort(x[top_ids])]         # x ↑   (changed)
    left_sorted   = left_ids[np.argsort(y[left_ids])]       # y ↑   (changed)

    def mk_segments(ids):
        if len(ids) < 2:
            return np.empty((0, 2), dtype=int)
        return np.column_stack([ids[:-1], ids[1:]])

    edges = {
        "bottom": {"nodes": bottom_sorted, "segments": mk_segments(bottom_sorted)},
        "right":  {"nodes": right_sorted,  "segments": mk_segments(right_sorted)},
        "top":    {"nodes": top_sorted,    "segments": mk_segments(top_sorted)},
        "left":   {"nodes": left_sorted,   "segments": mk_segments(left_sorted)},
    }
    return edges
Lx, Ly = 1.0, 1.0
cx, cy = 0.5, 0.5
r      = 0.20 * min(Lx, Ly)

Ntheta = 8
Nr     = 10


pts2d, quads = ogrid_plate_with_hole(Ntheta, Nr, Lx, Ly, cx, cy, r)
edges=order_outer_boundary(pts2d, Lx=1.0, Ly=1.0, atol=1e-8)


plt.figure(figsize=(6.5, 6.5))
for q in quads:
    poly = np.vstack([pts2d[q], pts2d[q[0]]])  # close loop
    plt.plot(poly[:, 0], poly[:, 1], 'k-', lw=0.3)


plt.plot([0, Lx, Lx, 0, 0], [0, 0, Ly, Ly, 0], 'k-', lw=1.0)

th = np.linspace(0, 2*np.pi, 400)
plt.plot(cx + r*np.cos(th), cy + r*np.sin(th), 'r-', lw=0.8)

plt.gca().set_aspect('equal', 'box')
plt.xlim(-0.02, Lx + 0.02); plt.ylim(-0.02, Ly + 0.02)
plt.title(f"Square with Circular Hole — QUAD4 (Nθ={Ntheta}, Nr={Nr})")
plt.xlabel("x"); plt.ylabel("y")
plt.tight_layout()
plt.show()
plt.show()
#%%
Xgp, J, detJ, dN_dx, Nvals, gp_parent = compute_q4_gauss_data(pts2d, quads)
node_data= compute_node_gauss_data(pts2d, quads)
Xgp_all, detJ_all, dNdx_all, valid_mask = pack_node_data(node_data)
rgt_bd_frc_w=np.vstack((rgt_bd_frc, np.zeros_like(rgt_bd_frc)))
lft_bd_frc_w=np.vstack((-lft_bd_frc, np.zeros_like(lft_bd_frc)))
top_bd_frc_w=np.vstack((np.zeros_like(top_bd_frc),top_bd_frc))
bot_bd_frc_w=np.vstack((np.zeros_like(bot_bd_frc),-bot_bd_frc))
print(rgt_bd_frc_w)
print(lft_bd_frc_w)
print(top_bd_frc_w)
print(bot_bd_frc_w)
tx_hist=[]
ty_hist=[]
meta_hist=[]
for i in range(len(t_hist)):
    reactions = {
        "left":   lft_bd_frc_w[:, i],   # [Fx_left,  Fy_left]
        "right":  rgt_bd_frc_w[:, i],
        "bottom": bot_bd_frc_w[:, i],
        "top":    top_bd_frc_w[:, i],
    }
    f1_edge, f2_edge = assemble_constant_from_reactions(pts2d, Lx, Ly, reactions, thickness=1.0)
    tx_hist.append(f1_edge); ty_hist.append(f2_edge)
tx_hist = jnp.stack([jnp.array(f) for f in tx_hist], axis=0)
ty_hist = jnp.stack([jnp.array(f) for f in ty_hist], axis=0)
#%% reading the image
image = Image.open('square_with_hole.png')
image = image.rotate(-90, expand=True)
img = np.array(image, dtype=np.float32)
if img.ndim == 3:
    img = img.mean(axis=2)  # grayscale
img /= img.max()
if not np.all((img == 0) | (img == 1)):
    raise ValueError("Image must contain only 0s and 1s.")
#%%
lr = 1e-4
gp_data_nodewise=[Xgp_all, detJ_all, dNdx_all, valid_mask ]
sharding = PositionalSharding(jax.devices()).reshape(n_cores, 1)
opt_init, opt_update, get_params = optimizers.adam(lr) 
opt_state = opt_init(Lambda_params)
node_X_ext = jnp.hstack([mesh.points, t_hist[-1]*np.ones_like(mesh.points[:,:1])])
NODE_w_unravel = lambda params: NODE_model_aniso(unravel(params))
Lambda_params, val_loss, metrics = train_colloc_parallel_weak( loss_weak_form,  [pts2d, t_hist],gp_data_nodewise,
    get_params, opt_update, opt_state,key,sharding=sharding,fname='cross',nIter=20000,print_freq=100,batch_size=200)

with open('hole_paper_post_weak_form_hyper_1_80_el_200samples_epo_20000.npy', 'wb') as f:
    pickle.dump([node_params, Lambda_params, val_loss, metrics, t_hist, lmb_hist], f)

# with open('hole_paper_post_weak_form_hyper_1_2560_el_500samples_epo_20000.npy', 'wb') as f:
#     pickle.dump([mesh, elem_X, strains, Force_x, Force_y, node_params, Lambda_params, val_loss, metrics, t_hist, lmb_hist], f)

