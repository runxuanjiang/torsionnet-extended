import numpy as np
import torch
import copy
import sys

from rdkit import Chem
from rdkit.Chem import AllChem

from conformer_rl import utils
from conformer_rl.agents import PPORecurrentAgent
from conformer_rl.config import Config, MolConfig
from conformer_rl.environments import Task
from conformer_rl.models import RTGNRecurrent
from conformer_rl.environments.environments import GibbsScorePruningEnv
from conformer_rl.environments.environment_components.action_mixins import DiscreteActionMixinMMFF

from conformer_rl.molecule_generation.generation.generate_lignin import generate_lignin

from gym.envs.registration import register

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

import logging
logging.basicConfig(level=logging.DEBUG)

class CustomEnv(DiscreteActionMixinMMFF, GibbsScorePruningEnv):
    pass

register(
    id = 'CustomConfEnv-v0',
    entry_point = '__main__:CustomEnv'
)

"""
    Change the 'mol_config.MMFFIters' parameter to test training for different values of MMFF max iterations.
"""

if __name__ == '__main__':
    utils.set_one_thread()

    mol = Chem.MolFromMolFile('4-lignin.mol')
    mol = Chem.AddHs(mol)
    AllChem.MMFFSanitizeMolecule(mol)
    AllChem.MMFFOptimizeMolecule(mol, maxIters=200)

    mol_config = MolConfig()
    mol_config.mol, mol_config.E0, mol_config.Z0 = mol, 209.55035793730713, 1.001529873400315
    mol_config.MMFFIters = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    mol_config_eval = copy.deepcopy(mol_config)
    mol_config_eval.MMFFIters = 1000

    print(f"Running 4-lignin baseline with {mol_config.MMFFIters} MMFF iters")

    config = Config()
    config.tag = f'4lignin-train_{mol_config.MMFFIters}_MMFFiters_'
    print(config.tag)
    config.network = RTGNRecurrent(6, 128, edge_dim=6, node_dim=5).to(device)
    # Batch Hyperparameters
    config.num_workers = 6
    config.rollout_length = 50
    config.recurrence = 2
    config.optimization_epochs = 4
    config.max_steps = 10000000
    config.save_interval = config.num_workers*200*5
    config.eval_interval = config.num_workers*200*5
    config.eval_episodes = 2
    config.mini_batch_size = 20

    # Coefficient Hyperparameters
    lr = 5e-6 * np.sqrt(config.num_workers)
    config.optimizer_fn = lambda params: torch.optim.Adam(params, lr=lr, eps=1e-5)

    # Task Settings
    config.train_env = Task('CustomConfEnv-v0', concurrency=True, num_envs=config.num_workers, seed=np.random.randint(0,1e5), mol_config=mol_config, max_steps=500)
    config.eval_env = Task('CustomConfEnv-v0', seed=np.random.randint(0,7e4), mol_config=mol_config_eval, max_steps=500)
    config.curriculum = None

    agent = PPORecurrentAgent(config)
    agent.run_steps()