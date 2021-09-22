import numpy as np
import torch
import time

from conformer_rl import utils
from conformer_rl.agents import PPORecurrentAgent
from conformer_rl.config import Config
from conformer_rl.environments import Task
from conformer_rl.models import RTGNRecurrent

from conformer_rl.molecule_generation import mol_from_molFile

from rdkit.Chem import AllChem

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

import logging
logging.basicConfig(level=logging.DEBUG)

if __name__ == '__main__':
    utils.set_one_thread()

    start_config = time.time()
    mol_config = mol_from_molFile('8monomers.mol', 1000)
    logging.debug(f'mol config time: {time.time() - start_config} seconds')

    config = Config()
    config.tag = '8_monomer_lignin'
    config.network = RTGNRecurrent(6, 128, edge_dim=6, node_dim=5).to(device)
    # Batch Hyperparameters
    config.num_workers = 20
    config.rollout_length = 20
    config.recurrence = 5
    config.optimization_epochs = 4
    config.max_steps = 10000000
    config.save_interval = config.num_workers*200*5
    config.eval_interval = config.num_workers*200*5
    config.eval_episodes = 2
    config.mini_batch_size = 50

    # Coefficient Hyperparameters
    lr = 5e-6 * np.sqrt(config.num_workers)
    config.optimizer_fn = lambda params: torch.optim.Adam(params, lr=lr, eps=1e-5)

    # Task Settings
    config.train_env = Task('GibbsScoreLogPruningEnv-v0', concurrency=True, num_envs=config.num_workers, seed=np.random.randint(0,1e5), mol_config=mol_config, max_steps=1000)
    config.eval_env = Task('GibbsScorePruningEnv-v0', seed=np.random.randint(0,7e4), mol_config=mol_config, max_steps=1000)
    config.curriculum = None

    print("Running 8 lignin")

    agent = PPORecurrentAgent(config)
    agent.run_steps()