import jax.numpy as jnp
import numpy as np
import jax
from jax import grad, vmap, jit, jacrev
from functools import partial
import jax.random as random
#from jax.experimental import optimizers
from jax.experimental.ode import odeint
import jax.example_libraries.optimizers as optimizers
from jax.scipy.optimize import minimize
from jax.lax import scan
from jax.nn import softplus
from jax.config import config
from jax.flatten_util import ravel_pytree

config.update("jax_enable_x64", True)
from jax.experimental.host_callback import id_print
rng = random.PRNGKey(2022)
import scipy



def init_layers(layers, key):
    Ws = []
    for i in range(len(layers) - 1):
        std_glorot = jnp.sqrt(2/(layers[i] + layers[i + 1]))
        key, subkey = random.split(key)
        Ws.append(random.normal(subkey, (layers[i], layers[i + 1]))*std_glorot)
    return Ws

def init_params(common_layers, sample_layers, key):
    params_I1_common = init_layers(common_layers, key)
    params_I1_sample = init_layers(sample_layers, key)
    params_I2_common = init_layers(common_layers, key)
    params_I2_sample = init_layers(sample_layers, key)
    # params_I1I2_common = init_layers(common_layers, key)
    # params_I1I2_sample = init_layers(sample_layers, key)

    params_I1 = (params_I1_common, params_I1_sample)
    params_I2 = (params_I2_common, params_I2_sample)
    # params_I1I2 = (params_I1I2_common, params_I1I2_sample)
    NODE_weights = (params_I1, params_I2)
    alpha = 0.5
    Psi1_bias = -5.0
    Psi2_bias = -5.0

    return (NODE_weights, Psi1_bias, Psi2_bias)

def init_params_aniso(common_layers, sample_layers, key):
    params_I1_common = init_layers(common_layers, key)
    params_I1_sample = init_layers(sample_layers, key)
    params_I2_common = init_layers(common_layers, key)
    params_I2_sample = init_layers(sample_layers, key)
    params_1_v_common = init_layers(common_layers, key)
    params_1_v_sample = init_layers(sample_layers, key)
    params_1_w_common = init_layers(common_layers, key)
    params_1_w_sample = init_layers(sample_layers, key)
    params_v_w_common = init_layers(common_layers, key)
    params_v_w_sample = init_layers(sample_layers, key)

    params_I1 = (params_I1_common, params_I1_sample)
    params_I2 = (params_I2_common, params_I2_sample)
    params_1_v = (params_1_v_common, params_1_v_sample)
    params_1_w = (params_1_w_common, params_1_w_sample)
    params_v_w = (params_v_w_common, params_v_w_sample)
    NODE_weights = (params_I1, params_I2, params_1_v, params_1_w, params_v_w)

    theta = 0.5
    Psi1_bias = -5.0
    Psi2_bias = -5.0
    alpha = [0.0, 0.0, 0.0]
    return (NODE_weights, theta, Psi1_bias, Psi2_bias, alpha)

@jit
def forward_pass(H, Ws):
    N_layers = len(Ws)
    for i in range(N_layers - 1):
        H = jnp.matmul(H, Ws[i])
        H = jnp.tanh(H)
    Y = jnp.matmul(H, Ws[-1])
    return Y

@jit
def common_forwardpass(X, params):
  common_params, sample_params = params
  # base net computation up to last layer
  H = forward_pass(X, common_params)
  Y = forward_pass(H, sample_params)
  return Y

#NODE forward pass
@jit
def NODE(y0, params, steps = 200):
    t0 = 0.0
    dt = 1.0/steps
    body_func = lambda y,t: (y + common_forwardpass(jnp.array([y]), params)[0]*dt, None)
    out, _ = scan(body_func, y0, jnp.linspace(0,1,steps), length = steps)
    return out
NODE_vmap = vmap(NODE, in_axes=(0, None), out_axes=0)

@jit
def RK_forwardpass(Y0, params):
  n = 4
  dt = 1.0/n
  def RK_step(Y,t):
    Y = jnp.array([Y])
    k1 = common_forwardpass(Y               , params)
    k2 = common_forwardpass(Y + 0.5*k1*dt   , params)
    k3 = common_forwardpass(Y + 0.5*k2*dt   , params)
    k4 = common_forwardpass(Y + k3*dt       , params)
    Y = Y + 1/6*dt*(k1 + 2*k2 + 2*k3 + k4)
    return (Y[0], None)
  out, _ = scan(RK_step, Y0, jnp.linspace(0,1,n), length = n)
  return out
RK_vmap = vmap(RK_forwardpass, in_axes=(0, None), out_axes=0)

# @jit
# def NODE(y0, params):
#     f = lambda y,t: common_forwardpass(jnp.array([y]), params)[0] # fake time argument for ODEint
#     return odeint(f, y0, jnp.array([0.0,1.0]), rtol=1.0e-4)[-1] # integrate between 0 and 1 and return the results at 1
# NODE_vmap = vmap(NODE, in_axes=(0, None), out_axes=0)


class NODE_model(): #isotropic
    def __init__(self, params):
        NODE_weights, self.Psi1_bias, self.Psi2_bias = params
        self.params_I1, self.params_I2 = NODE_weights
        
    def Psi1(self, I1, I2):
        I1 = I1-3.0
        I2 = I2-3.0
        Psi_1 = NODE(I1, self.params_I1)
        # a = jax.nn.sigmoid(self.alpha)
        # Psi_1_2 = NODE(a*I1 + (1.0-a)*I2, self.params_1_2)
        Psi_1_2 = a = 0.0
        return Psi_1 + a*Psi_1_2 + jnp.exp(self.Psi1_bias)
    
    def Psi2(self, I1, I2):
        I1 = I1-3.0
        I2 = I2-3.0
        Psi_2 = NODE(I2, self.params_I2)
        # a = jax.nn.sigmoid(self.alpha)
        # Psi_1_2 = NODE(a*I1 + (1.0-a)*I2, self.params_1_2)
        Psi_1_2 = a = 0.0
        return Psi_2 + (1.0-a)*Psi_1_2 + jnp.exp(self.Psi2_bias)
    

class NODE_model_aniso(): #anisotropic

    def __init__(self, params):
        NODE_weights, self.theta, self.Psi1_bias, self.Psi2_bias, self.alpha = params
        self.params_I1, self.params_I2, self.params_1_v, self.params_1_w, self.params_v_w = NODE_weights
    
    def Psi1(self, I1, I2, Iv, Iw):
        I1 = I1-3.0
        Iv = Iv-1.0
        Iw = Iw-1.0
        Psi_1 = RK_forwardpass(I1, self.params_I1)
        a = jax.nn.sigmoid(self.alpha[0])
        Psi_v_1 = RK_forwardpass(a*I1 + (1.0-a)*Iv, self.params_1_v)*a
        Psi_v_1 = jnp.maximum(Psi_v_1, 0.0)
        a = jax.nn.sigmoid(self.alpha[1])
        Psi_w_1 = RK_forwardpass(a*I1 + (1.0-a)*Iw, self.params_1_w)*a
        Psi_w_1 = jnp.maximum(Psi_w_1, 0.0)
        return Psi_1 + Psi_v_1 + Psi_w_1 + jnp.exp(self.Psi1_bias)
    
    def Psi2(self, I1, I2, Iv, Iw):
        I2 = I2-3.0
        Psi_2 = RK_forwardpass(I2, self.params_I2)
        return Psi_2 + jnp.exp(self.Psi2_bias)
    
    def Psiv(self, I1, I2, Iv, Iw):
        I1 = I1-3.0
        Iv = Iv-1.0
        Iw = Iw-1.0
        a = jax.nn.sigmoid(self.alpha[0])
        Psi_1_v = RK_forwardpass(a*I1 + (1.0-a)*Iv, self.params_1_v)*(1.0-a)
        Psi_1_v = jnp.maximum(Psi_1_v, 0.0)
        a = jax.nn.sigmoid(self.alpha[2])
        Psi_w_v = RK_forwardpass(a*Iv + (1.0-a)*Iw, self.params_v_w)*a
        Psi_w_v = jnp.maximum(Psi_w_v, 0.0)
        return Psi_1_v + Psi_w_v
    
    def Psiw(self, I1, I2, Iv, Iw):
        I1 = I1-3.0
        Iv = Iv-1.0
        Iw = Iw-1.0
        a = jax.nn.sigmoid(self.alpha[1])
        Psi_1_w = RK_forwardpass(a*I1 + (1.0-a)*Iw, self.params_1_w)*(1.0-a)
        Psi_1_w = jnp.maximum(Psi_1_w, 0.0)
        a = jax.nn.sigmoid(self.alpha[2])
        Psi_v_w = RK_forwardpass(a*Iv + (1.0-a)*Iw, self.params_v_w)*(1.0-a)
        Psi_v_w = jnp.maximum(Psi_v_w, 0.0)
        return Psi_1_w + Psi_v_w

class GOH_model(): #anisotropic
    def __init__(self, params):
        self.params = params
        self.theta = params[-1]
    
    def Psi1(self, I1, I2, Iv, Iw):
        C10, k1, k2, kappa, _ = self.params

        E = kappa*(I1-3.0) + (1-3*kappa)*(Iv-1.0)
        E = jnp.maximum(E, 0.0)
        Psi1 = C10 + k1*kappa*E*jnp.exp(k2*E**2)
        return Psi1
    
    def Psi2(self, I1, I2, Iv, Iw):
        return 0.0
    
    def Psiv(self, I1, I2, Iv, Iw):
        C10, k1, k2, kappa, _ = self.params

        E = kappa*(I1-3.0) + (1-3*kappa)*(Iv-1.0)
        E = jnp.maximum(E, 0.0)
        Psiv = k1*(1-3*kappa)*E*jnp.exp(k2*E**2)
        return Psiv
    
    def Psiw(self, I1, I2, Iv, Iw):
        return 0.0
    
class HGO_model():
    def __init__(self, params):
        C10, k1_v, k1_w, k2_v, k2_w, theta = params
        C10, k1_v, k1_w, k2_v, k2_w = jnp.exp(C10), jnp.exp(k1_v), jnp.exp(k1_w), jnp.exp(k2_v), jnp.exp(k2_w)
        self.params = [C10, k1_v, k1_w, k2_v, k2_w, theta]
        self.theta = params[-1]

    def Psi1(self, I1, I2, Iv, Iw):
        C10, k1_v, k1_w, k2_v, k2_w, theta = self.params
        return C10
    
    def Psi2(self, I1, I2, Iv, Iw):
        return 0.0
    
    def Psiv(self, I1, I2, Iv, Iw):
        C10, k1_v, k1_w, k2_v, k2_w, theta = self.params
        Iv = jnp.maximum(Iv, 1.0)
        return k1_v*(Iv-1.0)*jnp.exp(k2_v*(Iv-1)**2)
    
    def Psiw(self, I1, I2, Iv, Iw):
        C10, k1_v, k1_w, k2_v, k2_w, theta = self.params
        Iw = jnp.maximum(Iw, 1.0)
        return k1_w*(Iw-1.0)*jnp.exp(k2_w*(Iw-1)**2)

class neohook_model():
    def __init__(self, C10):
        self.C10 = C10
        self.theta = 0.0

    def Psi1(self, I1, I2, Iv, Iw):
        return self.C10
    
    def Psi2(self, I1, I2, Iv, Iw):
        return 0.0
    
    def Psiv(self, I1, I2, Iv, Iw):
        return 0.0
    
    def Psiw(self, I1, I2, Iv, Iw):
        return 0.0



def split_c_s_params(node_params):
    NODE_weights, theta, Psi1_bias, Psi2_bias, alpha = node_params
    params_I1, params_I2, params_1_v, params_1_w, params_v_w = NODE_weights
    params_I1_common, params_I1_sample = params_I1
    params_I2_common, params_I2_sample = params_I2
    params_1_v_common, params_1_v_sample = params_1_v
    params_1_w_common, params_1_w_sample = params_1_w
    params_v_w_common, params_v_w_sample = params_v_w
    sample_params = (params_I1_sample, params_I2_sample, params_1_v_sample, params_1_w_sample, params_v_w_sample, theta, Psi1_bias, Psi2_bias, alpha)
    common_params = (params_I1_common, params_I2_common, params_1_v_common, params_1_w_common, params_v_w_common)
    return common_params, sample_params

def merge_c_s_params(common_params, sample_params):
    params_I1_sample, params_I2_sample, params_1_v_sample, params_1_w_sample, params_v_w_sample, theta, Psi1_bias, Psi2_bias, alpha = sample_params
    params_I1_common, params_I2_common, params_1_v_common, params_1_w_common, params_v_w_common = common_params
    params_I1 = (params_I1_common, params_I1_sample)
    params_I2 = (params_I2_common, params_I2_sample)
    params_1_v = (params_1_v_common, params_1_v_sample)
    params_1_w = (params_1_w_common, params_1_w_sample)
    params_v_w = (params_v_w_common, params_v_w_sample)
    NODE_weights = (params_I1, params_I2, params_1_v, params_1_w, params_v_w)
    node_params = (NODE_weights, theta, Psi1_bias, Psi2_bias, alpha)
    return node_params



def eval_Cauchy_iso(lmbx,lmby, model):
    lmbz = 1.0/(lmbx*lmby)
    F = jnp.array([[lmbx, 0, 0],
                   [0, lmby, 0],
                   [0, 0, lmbz]])
    C = F.T @ F
    C2 = C @ C
    Cinv = jnp.linalg.inv(C)
    I1 = C[0,0] + C[1,1] + C[2,2]
    trC2 = C2[0,0] + C2[1,1] + C2[2,2]
    I2 = 0.5*(I1**2 - trC2)

    Psi1 = model.Psi1(I1, I2)
    Psi2 = model.Psi2(I1, I2)
    
    p = -C[2,2]*(2*Psi1 + 2*Psi2*(I1 - C[2,2]))
    S = p*Cinv + 2*Psi1*jnp.eye(3) + 2*Psi2*(I1*jnp.eye(3)-C)
    sgm = F @ (S @ F.T)
    return sgm
eval_Cauchy_iso_vmap = vmap(eval_Cauchy_iso, in_axes=(0,0,None), out_axes = 0)

def eval_P(F_xx, F_xy, F_yx, F_yy, model):
    F = jnp.array([[F_xx, F_xy, 0.0],
                   [F_yx, F_yy, 0.0],
                   [0.0, 0.0, 1/(F_xx*F_yy-F_xy*F_yx)]])
    C = F.T @ F

    C2 = C @ C
    Cinv = jnp.linalg.inv(C)
    theta = model.theta
    v0 = jnp.array([ jnp.cos(theta), jnp.sin(theta), 0])
    w0 = jnp.array([-jnp.sin(theta), jnp.cos(theta), 0])
    V0 = jnp.outer(v0, v0)
    W0 = jnp.outer(w0, w0)

    I1 = C[0,0] + C[1,1] + C[2,2]
    trC2 = C2[0,0] + C2[1,1] + C2[2,2]
    I2 = 0.5*(I1**2 - trC2)
    Iv = jnp.einsum('ij,ij',C,V0)
    Iw = jnp.einsum('ij,ij',C,W0)

    Psi1 = model.Psi1(I1, I2, Iv, Iw)
    Psi2 = model.Psi2(I1, I2, Iv, Iw)
    Psiv = model.Psiv(I1, I2, Iv, Iw)
    Psiw = model.Psiw(I1, I2, Iv, Iw)

    p = -C[2,2]*(2*Psi1 + 2*Psi2*(I1 - C[2,2]) + 2*Psiv*V0[2,2] + 2*Psiw*W0[2,2])
    S = p*Cinv + 2*Psi1*jnp.eye(3) + 2*Psi2*(I1*jnp.eye(3)-C) + 2*Psiv*V0 + 2*Psiw*W0

    P = F @ S
    return P
eval_P_vmap = vmap(eval_P, in_axes=(0,0,0,0,None), out_axes = 0)

def eval_Cauchy(F_xx, F_xy, F_yx, F_yy, model):
    F = jnp.array([[F_xx, F_xy, 0.0],
                   [F_yx, F_yy, 0.0],
                   [0.0, 0.0, 1/(F_xx*F_yy-F_xy*F_yx)]])
    P = eval_P(F_xx, F_xy, F_yx, F_yy, model)
    sgm = P @ F.T
    return sgm
eval_Cauchy_vmap = vmap(eval_Cauchy, in_axes=(0,0,0,0,None), out_axes = 0)

def merge_weights_aniso(params_c, params_s):
    NODE_weights, theta, Psi1_bias, Psi2_bias, alpha = params_c
    params_I1, params_I2, params_1_v, params_1_w, params_v_w = NODE_weights
    params_I1c,params_I1s = params_I1
    params_I2c,params_I2s = params_I2
    params_1_vc,params_1_vs = params_1_v
    params_1_wc,params_1_ws = params_1_w
    params_v_wc,params_v_ws = params_v_w

    params_I1   = (params_I1c,  params_s[0])
    params_I2   = (params_I2c,  params_s[1])
    params_1_v  = (params_1_vc, params_s[2])
    params_1_w  = (params_1_wc, params_s[3])
    params_v_w  = (params_v_wc, params_s[4])
    NODE_weights = (params_I1, params_I2, params_1_v, params_1_w, params_v_w)
    params = [NODE_weights, params_s[5], params_s[6], params_s[7], params_s[8]]
    return params


class ThreeDElasticity():
    def __init__(self, model):
        self.model = model

    def ugrad_2_P(self, u_grad, params, dim):
        theta = 0.0
        #Kinematics
        F_2D = u_grad + jnp.eye(dim)
        C_2D = F_2D.T @ F_2D
        Cinv_2D = jnp.linalg.inv(C_2D)
        detC_2D = jnp.linalg.det(C_2D)
        C33 = 1/detC_2D
        C = jnp.array([  [C_2D[0,0], C_2D[0,1], 0],\
                        [C_2D[1,0], C_2D[1,1], 0],\
                        [0,         0,       C33]])
        C2 = C @ C
        Cinv = jnp.linalg.inv(C)
        I1 = C[0,0] + C[1,1] + C[2,2]
        trC2 = C2[0,0] + C2[1,1] + C2[2,2]
        I2 = 0.5*(I1**2 - trC2)
        v0 = jnp.array([ jnp.cos(theta), jnp.sin(theta), 0])
        w0 = jnp.array([-jnp.sin(theta), jnp.cos(theta), 0])
        V0 = jnp.outer(v0, v0)
        W0 = jnp.outer(w0, w0)
        Iv = jnp.einsum('ij,ij',C,V0)
        Iw = jnp.einsum('ij,ij',C,W0)

        # Energy/Stress
        model = self.model(params)
        Psi1 = model.Psi1(I1,I2,Iv,Iw)
        Psi2 = model.Psi2(I1,I2,Iv,Iw)
        Psiv = model.Psiv(I1,I2,Iv,Iw)
        Psiw = model.Psiw(I1,I2,Iv,Iw)
        
        p = -C[2,2]*(2*Psi1 + 2*Psi2*(I1 - C[2,2]) + 2*Psiv*V0[2,2] + 2*Psiw*W0[2,2])
        S = p*Cinv + 2*Psi1*jnp.eye(3) + 2*Psi2*(I1*jnp.eye(3)-C) + 2*Psiv*V0 + 2*Psiw*W0
        S_2D = S[:2,:2]
        P_2D = F_2D @ S_2D
        return P_2D