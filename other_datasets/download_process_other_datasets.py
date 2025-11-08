import os, sys
import json
import re
import pandas as pd

# Try to import necessary libraries and provide helpful error messages.
try:
    from matminer.datasets import load_dataset
    from pymatgen.core import Structure, Lattice, Element
except ImportError as e:
    print(f"Error: A required library is missing. {e}", file=sys.stderr)
    print("Please install required libraries using: pip install matminer pandas", file=sys.stderr)
    exit(1)

# --- Structure Generation Functions from Previous Scripts ---

def get_m2ax_structure(row):
    """Generates a M2AX structure."""
    formula = row['formula']
    a, c, d_mx = float(row['a']), float(row['c']), float(row['d_mx'])
    
    # Parse M, A, X from formula like 'Sc2AlC'
    parts = re.findall(r'([A-Z][a-z]*)(\d*)', formula)
    m_element, a_element, x_element = parts[0][0], parts[1][0], parts[2][0]

    # Calculate zM using the convention from our discussion
    zm = 0.5 - (d_mx / c)
    
    lattice = Lattice.hexagonal(a, c)
    species = [m_element, a_element, x_element]
    coords = [[1/3, 2/3, zm], [1/3, 2/3, 3/4], [0, 0, 0]]
    
    return Structure.from_spacegroup(194, lattice, species, coords)

def get_perovskite_structure(row):
    """Generates a perovskite structure using the unified AAO3/ABO3 model."""
    formula = row['formula']
    a, b, c = float(row['a']), float(row['b']), float(row['c'])
    alpha, beta, gamma = float(row['alpha']), float(row['beta']), float(row['gamma'])

    cation_part = formula.replace('O3', '')
    parts = re.findall(r'([A-Z][a-z]*)(\d*)', cation_part)
    atom_a, atom_b = (parts[0][0], parts[0][0]) if len(parts) == 1 else (parts[0][0], parts[1][0])

    lattice = Lattice.from_parameters(a, b, c, alpha, beta, gamma)
    species = [atom_a, atom_b, 'O', 'O', 'O']
    coords = [
        [0.0, 0.0, 0.0], [0.5, 0.5, 0.5], [0.5, 0.5, 0.0],
        [0.5, 0.0, 0.5], [0.0, 0.5, 0.5]
    ]
    return Structure(lattice, species, coords)

def get_double_perovskite_structure(row):
    """Generates a fully ordered, intercalated, cubic double perovskite."""
    a1, a2 = row['a_1'], row['a_2']
    b1, b2 = row['b_1'], row['b_2']

    prim_lattice = Lattice.cubic(4.25)
    prim_species = ['A', 'B', 'O', 'O', 'O']
    prim_coords = [
        [0.0, 0.0, 0.0], [0.5, 0.5, 0.5], [0.5, 0.5, 0.0],
        [0.5, 0.0, 0.5], [0.0, 0.5, 0.5]
    ]
    structure = Structure(prim_lattice, prim_species, prim_coords)
    structure.make_supercell([2, 2, 2])

    final_species = []
    for site in structure:
        prim_cell_x = int(site.frac_coords[0] * 2)
        prim_cell_y = int(site.frac_coords[1] * 2)
        prim_cell_z = int(site.frac_coords[2] * 2)
        parity = (prim_cell_x + prim_cell_y + prim_cell_z) % 2
        
        if site.specie.symbol == 'A':
            final_species.append(a1 if parity == 0 else a2)
        elif site.specie.symbol == 'B':
            final_species.append(b1 if parity == 0 else b2)
        else:
            final_species.append('O')
            
    final_structure = Structure(structure.lattice, final_species, structure.frac_coords)
    final_structure.remove_oxidation_states()
    return final_structure

# --- Main Script ---

def main():
    datasets = [
        "tholander_nitrides",
        "boltztrap_mp",
      #   "m2ax",
      #   "wolverton_oxides",
      #   "double_perovskites_gap"
    ]
    output_folder = "augmented_datasets"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Created output folder: {output_folder}")

    for name in datasets:
        print(f"\n--- Processing dataset: {name} ---")
        try:
            df = load_dataset(name)
            df.columns = df.columns.str.strip()
            print(f"Successfully loaded raw data for {name}.")

            if name == "tholander_nitrides":
                df['initial_structure_json'] = df['initial_structure'].apply(lambda s: json.dumps(s.as_dict()) if s else None)
                df['final_structure_json'] = df['final_structure'].apply(lambda s: json.dumps(s.as_dict()) if s else None)
                df = df.drop(columns=['initial_structure', 'final_structure'])
            
            elif name == "boltztrap_mp":
                df['structure_json'] = df['structure'].apply(lambda s: json.dumps(s.as_dict()) if s else None)
                df = df.drop(columns=['structure'])

            elif name == "m2ax":
                df['pymatgen_structure_json'] = df.apply(get_m2ax_structure, axis=1).apply(lambda s: json.dumps(s.as_dict()))

            elif name == "wolverton_oxides":
                df['pymatgen_structure_json'] = df.apply(get_perovskite_structure, axis=1).apply(lambda s: json.dumps(s.as_dict()))
            
            elif name == "double_perovskites_gap":
                df['pymatgen_structure_json'] = df.apply(get_double_perovskite_structure, axis=1).apply(lambda s: json.dumps(s.as_dict()))

            output_path = os.path.join(output_folder, f"{name}_augmented.csv")
            df.to_csv(output_path, index=False)
            print(f"Successfully processed and saved augmented data to '{output_path}'")

        except Exception as e:
            print(f"Failed to process dataset {name}. Error: {e}", file=sys.stderr)

if __name__ == "__main__":
    main()
