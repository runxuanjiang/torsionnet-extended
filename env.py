import numpy as np
import torch

from rdkit import Chem

from torch_geometric.data import Data, Batch
from torch_geometric.transforms import Distance, NormalizeScale, Center, NormalizeRotation

from torsionnet.environments.conformer_environment import ConformerEnv

from torsionnet.environments.conformer_environment.obs_mixins import bond_features, get_bond_pair, atom_features

from molecule import BUILDING_BLOCKS, BUILDING_BLOCK_MAP, get_building_block_id_from_torsion, get_torsion_building_block_atoms, get_building_block_torsion_id

ACTION_MAP = [[0, 1], [3, 2], [4, 5], [7, 6]]

def mol_to_graph(molecule):
        mol = Chem.RemoveHs(molecule)
        conf = mol.GetConformer(id=-1)
        atoms = mol.GetAtoms()
        bonds = mol.GetBonds()
        node_f = [atom_features(atom, conf) for atom in atoms]
        edge_index = get_bond_pair(mol)
        edge_attr = [bond_features(bond) for bond in bonds]
        for bond in bonds:
            edge_attr.append(bond_features(bond))

        data = Data(
                    x=torch.tensor(node_f, dtype=torch.float),
                    edge_index=torch.tensor(edge_index, dtype=torch.long),
                    edge_attr=torch.tensor(edge_attr,dtype=torch.float),
                    pos=torch.Tensor(conf.GetPositions())
                )

        data = Center()(data)
        data = NormalizeRotation()(data)
        data.x[:,-3:] = data.pos
        data = Batch.from_data_list([data])
        return data


class XorgateSkeletonPointsObsMixin(ConformerEnv):

    def _get_obs(self):
        if not hasattr(self, 'action'):
            self.action = [0, 0, 0, 0, 0, 0, 0, 0]
        main_mol = self.molecule
        main_graph = mol_to_graph(main_mol)

        bb_graphs = []


        for bb_idx in BUILDING_BLOCK_MAP:
            bb_mol = BUILDING_BLOCKS[bb_idx]
            bb_conf = bb_mol.GetConformer(id=0)
            nonring, _ = Chem.TorsionFingerprints.CalculateTorsionLists(bb_mol)
            nonring = [list(atoms[0]) for atoms, _ in nonring]
            for i, tors in enumerate(nonring):
                ang = -180 + 60 * self.action[ACTION_MAP[bb_idx][i]]
                Chem.rdMolTransforms.SetDihedralDeg(bb_conf, tors[0], tors[1], tors[2], tors[3], float(ang))
            Chem.AllChem.MMFFOptimizeMolecule(bb_mol, maxIters=1000, confId=0)

            bb_graphs.append(mol_to_graph(bb_mol))

        import pdb
        pdb.set_trace()

        return main_graph, bb_graphs, self.nonring, 

from torsionnet.environments.conformer_environment import GibbsRewardMixin, UniqueGibbsRewardMixin, PruningGibbsRewardMixin
from torsionnet.environments.conformer_environment import DiscreteActionMixin

class XorgateHierarchicalEnv(PruningGibbsRewardMixin, DiscreteActionMixin, XorgateSkeletonPointsObsMixin):
    pass


from gym.envs.registration import register

register(
    id='XorgateEnv-v0',
    entry_point='env:XorgateHierarchicalEnv'
)
