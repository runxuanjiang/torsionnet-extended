from conformer_rl.molecule_generation.molecules import mol_from_molFile

config = mol_from_molFile('12monomers.mol', 1000)
print(config.E0, config.Z0)