import jax.numpy as jnp

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
        return k1_v*(Iv-1.0)*jnp.exp(k2_v*(Iv-1)**2)
    
    def Psiw(self, I1, I2, Iv, Iw):
        C10, k1_v, k1_w, k2_v, k2_w, theta = self.params
        return k1_w*(Iw-1.0)*jnp.exp(k2_w*(Iw-1)**2)