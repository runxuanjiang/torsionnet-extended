import numpy as np
import torch

from rdkit import Chem

from torch_geometric.data import Data, Batch
from torch_geometric.transforms import Distance, NormalizeScale, Center, NormalizeRotation

from torsionnet.environments.conformer_environment import ConformerEnv

from .molecule import BUILDING_BLOCKS, BUILDING_BLOCK_MAP, get_building_block_id_from_torsion, get_torsion_building_block_atoms, get_building_block_torsion_id


class XorgateSkeletonPointsObsMixin(ConformerEnv):

    def _get_obs(self):
        


from torsionnet.environments.conformer_environment import GibbsRewardMixin, UniqueGibbsRewardMixin, PruningGibbsRewardMixin
from torsionnet.environments.conformer_environment import DiscreteActionMixin

class XorgateHierarchicalEnv(PruningGibbsRewardMixin, DiscreteActionMixin, XorgateSkeletonPointsObsMixin):
    pass


from gym.envs.registration import register

register(
    id='XorgateEnv-v0',
    entry_point='env:XorgateHierarchicalEnv'
)
