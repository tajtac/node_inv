import os
n_cores = 20
os.environ["XLA_FLAGS"] = '--xla_force_host_platform_device_count={}'.format(n_cores)

from jax.config import config
config.update("jax_enable_x64", True)

import jax
import jax.numpy as jnp
import pickle
from jax import jit
import jax.random as random
key = random.PRNGKey(2022)
import jax.example_libraries.optimizers as optimizers
from jax.flatten_util import ravel_pytree
from jax.sharding import PositionalSharding

from utils_node import NODE_model_aniso
from fem import write_biax_abaqus_inp
from utils import train_colloc_parallel as train, ff_nn, divergence, bd_forces, a1, a2, lr
import argparse



# Load the pretraining data
def retrain_inverse(fname, nIter=20_000):
    with open('params/'+fname+'_pre.npy', 'rb') as f:
        coord_2_strain_params, node_params, Lambda_params, node_X, elements, n_node, n_elem, Fx, Fy, strains, \
            rgt_bd_X, top_bd_X, lft_bd_X, bot_bd_X = pickle.load(f)
    _, unravel = ravel_pytree(node_params)
    NODE_w_unravel = lambda params: NODE_model_aniso(unravel(params))
    bd_X = rgt_bd_X, top_bd_X, lft_bd_X, bot_bd_X

    # Train
    @jit
    def loss(Lambda_params, X_colloc):
        div_x, div_y = divergence(Lambda_params, X_colloc, bd_X, coord_2_strain_params, NODE_w_unravel)
        F_rgt, F_top, F_lft, F_bot = bd_forces(Lambda_params, None, bd_X, coord_2_strain_params, NODE_w_unravel)
        return a1*(div_x**2 + div_y**2) + a2*((F_rgt-Fx)**2 + (F_top-Fy)**2 + (F_lft-Fx)**2 + (F_bot-Fy)**2)

    sharding = PositionalSharding(jax.devices()).reshape(n_cores, 1)
    opt_init, opt_update, get_params = optimizers.adam(lr) 
    opt_state = opt_init(Lambda_params)

    metric1 = lambda params, X:  bd_forces(params, X, bd_X, coord_2_strain_params, NODE_w_unravel)
    metric2 = lambda params, X: divergence(params, X, bd_X, coord_2_strain_params, NODE_w_unravel)

    Lambda_params, val_loss, metrics = train(loss, node_X, get_params, opt_update, opt_state, 
                                                key, sharding, fname, nIter = nIter, print_freq=1000, metric_fns=[metric1, metric2])

    with open('params/'+fname+'_post.npy', 'wb') as f:
        pickle.dump([node_X, strains, Fx, Fy, node_params, Lambda_params, val_loss, metrics], f)

    # Write abaqus
    lmx = 1.1
    lmy = 1.1
    write_biax_abaqus_inp(Lambda_params, ff_nn, node_params, lmx-1, lmy-1, outputfile='abaqus/'+fname+'.inp', disp_or_force='disp', inputfile='abaqus/equi_strain.inp')



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Retraining inverse problem")
    parser.add_argument("--fname", type=str, required=True)
    parser.add_argument("--nIter", type=int)
    args, args_other = parser.parse_known_args()


    retrain_inverse(
        fname=args.fname,
        nIter=args.nIter,
    )
