from rdkit import Chem
from rdkit.Chem import AllChem
from torsionnet.generate_molecule.xor_gate import XorGate
from torsionnet.generate_molecule.molecule_wrapper import MoleculeWrapper

xorgate = XorGate(gate_complexity=2, num_gates=4)
xorgate = xorgate.polymer.stk_molecule

building_block = list(xorgate.get_building_blocks())[0].to_rdkit_mol()
AllChem.EmbedMolecule(building_block, randomSeed=0)
BUILDING_BLOCKS = [building_block]
BUILDING_BLOCK_MAP = [0, 0, 0, 0]

def get_building_block_torsion_id(torsion):
    building_block_atoms = get_torsion_building_block_atoms(torsion)
    building_block_id = get_building_block_id_from_torsion(torsion)
    building_block_mol = BUILDING_BLOCKS[BUILDING_BLOCK_MAP[building_block_id]]
    nonring, _ = Chem.TorsionFingerprints.CalculateTorsionLists(building_block_mol)
    nonring = [list(atoms[0]) for atoms, ang in nonring]

    res = 0
    for i, atoms in enumerate(nonring):
        count = 0
        for atom in building_block_atoms:
            if atom in atoms:
                count += 1
        if count >= 2:
            res = i
            break
    return res

def get_building_block_id_from_torsion(torsion):
    seen_building_blocks = {}
    for atom_info in xorgate.get_atom_infos(torsion):
        building_block_id = atom_info.get_building_block_id()
        if building_block_id in seen_building_blocks:
            seen_building_blocks[building_block_id] += 1
        else:
            seen_building_blocks[building_block_id] = 1

    return max(seen_building_blocks, key=seen_building_blocks.get)

def get_torsion_building_block_atoms(torsion):
    building_block_atoms = []
    for atom_info in xorgate.get_atom_infos(torsion):
        building_block_atoms.append(atom_info.get_building_block_atom().get_id())
    return building_block_atoms


XORGATE = [MoleculeWrapper(xorgate.to_rdkit_mol(),
standard=270,
input_type='mol')]