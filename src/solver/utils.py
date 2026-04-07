import copy
import json
import math
import numpy as np
import pandas as pd
import torch
from scipy.spatial import cKDTree
from rdkit import Chem
from rdkit.Chem import RWMol
from rdkit.Chem import Draw, AllChem
from rdkit.Chem import rdDepictor
import matplotlib.pyplot as plt
import re
from typing import List
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
ELEMENTS = [
    "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne",
    "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar", "K", "Ca",
    "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
    "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr", "Y", "Zr",
    "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn",
    "Sb", "Te", "I", "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd",
    "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb",
    "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
    "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th",
    "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm",
    "Md", "No", "Lr", "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds",
    "Rg", "Cn", "Nh", "Fl", "Mc", "Lv", "Ts", "Og"
]
COLORS = {
    u'c': '0.0,0.75,0.75', u'b': '0.0,0.0,1.0', u'g': '0.0,0.5,0.0', u'y': '0.75,0.75,0',
    u'k': '0.0,0.0,0.0', u'r': '1.0,0.0,0.0', u'm': '0.75,0,0.75'
}
RGROUP_SYMBOLS = ['R', 'R1', 'R2', 'R3', 'R4', 'R5', 'R6', 'R7', 'R8', 'R9', 'R10', 'R11', 'R12',
                  'Ra', 'Rb', 'Rc', 'Rd', 'X', 'Y', 'Z', 'Q', 'A', 'E', 'Ar']
VALENCES = {
    "H": [1], "Li": [1], "Be": [2], "B": [3], "C": [4], "N": [3, 5], "O": [2], "F": [1],
    "Na": [1], "Mg": [2], "Al": [3], "Si": [4], "P": [5, 3], "S": [6, 2, 4], "Cl": [1], "K": [1], "Ca": [2],
    "Br": [1], "I": [1], "*":[3,4,5,6], 
}
def adjust_bbox1(large_bbox, small_bbox, bond_bbox):
    x_min_l, y_min_l, x_max_l, y_max_l = large_bbox
    x_min_s, y_min_s, x_max_s, y_max_s = small_bbox
    x_min_b, y_min_b, x_max_b, y_max_b = bond_bbox
    scaled_box= max([x_min_l,x_min_s,x_min_b]),max([y_min_l,y_min_s,y_min_b]),x_max_l, y_max_l
    return large_bbox
def view_box_center(bond_bbox,heavy_centers):
    fig, ax = plt.subplots(figsize=(10, 10))
    for box in bond_bbox:
        x1, y1, x2, y2 = box
        width = x2 - x1
        height = y2 - y1
        rect = Rectangle((x1, y1), width, height, linewidth=1, edgecolor='blue', facecolor='none')
        ax.add_patch(rect)
    for center in heavy_centers:
        x, y = center
        circle = Circle((x, y), radius=5, edgecolor='red', facecolor='none', linewidth=1)
        ax.add_patch(circle)
    x_min = min(bond_bbox[:, 0].min(), heavy_centers[:, 0].min()) - 10
    x_max = max(bond_bbox[:, 2].max(), heavy_centers[:, 0].max()) + 10
    y_min = min(bond_bbox[:, 1].min(), heavy_centers[:, 1].min()) - 10
    y_max = max(bond_bbox[:, 3].max(), heavy_centers[:, 1].max()) + 10
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_title("Boxes and Centers")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    plt.gca().set_aspect('equal', adjustable='box')  
    plt.grid(True, linestyle='--', alpha=0.7)
def molIDX(mol):
    for i, atom in enumerate(mol.GetAtoms()):
        atom.SetAtomMapNum(i)  
    return mol
def molIDX_del(mol):
    for i, atom in enumerate(mol.GetAtoms()):
        atom.SetAtomMapNum(0)  
        print(i)
    return mol
class Substitution(object):
    def __init__(self, abbrvs, smarts, smiles, probability):
        assert type(abbrvs) is list
        self.abbrvs = abbrvs
        self.smarts = smarts
        self.smiles = smiles
        self.probability = probability
SUBSTITUTIONS: List[Substitution] = [
    Substitution(['NO2', 'O2N'], '[N+](=O)[O-]', "[N+](=O)[O-]", 0.5),
    Substitution(['CHO', 'OHC'], '[CH1](=O)', "[CH1](=O)", 0.5),
    Substitution(['CO2Et', 'COOEt'], 'C(=O)[OH0;D2][CH2;D2][CH3]', "[C](=O)OCC", 0.5),
    Substitution(['OAc'], '[OH0;X2]C(=O)[CH3]', "[O]C(=O)C", 0.7),
    Substitution(['NHAc'], '[NH1;D2]C(=O)[CH3]', "[NH]C(=O)C", 0.7),
    Substitution(['Ac'], 'C(=O)[CH3]', "[C](=O)C", 0.1),
    Substitution(['OBz'], '[OH0;D2]C(=O)[cH0]1[cH][cH][cH][cH][cH]1', "[O]C(=O)c1ccccc1", 0.7),  
    Substitution(['Bz'], 'C(=O)[cH0]1[cH][cH][cH][cH][cH]1', "[C](=O)c1ccccc1", 0.2),  
    Substitution(['OBn'], '[OH0;D2][CH2;D2][cH0]1[cH][cH][cH][cH][cH]1', "[O]Cc1ccccc1", 0.7),  
    Substitution(['Bn'], '[CH2;D2][cH0]1[cH][cH][cH][cH][cH]1', "[CH2]c1ccccc1", 0.2),  
    Substitution(['NHBoc'], '[NH1;D2]C(=O)OC([CH3])([CH3])[CH3]', "[NH1]C(=O)OC(C)(C)C", 0.6),
    Substitution(['NBoc'], '[NH0;D3]C(=O)OC([CH3])([CH3])[CH3]', "[NH1]C(=O)OC(C)(C)C", 0.6),
    Substitution(['Boc'], 'C(=O)OC([CH3])([CH3])[CH3]', "[C](=O)OC(C)(C)C", 0.2),
    Substitution(['Cbm'], 'C(=O)[NH2;D1]', "[C](=O)N", 0.2),
    Substitution(['Cbz'], 'C(=O)OC[cH]1[cH][cH][cH1][cH][cH]1', "[C](=O)OCc1ccccc1", 0.4),
    Substitution(['Cy'], '[CH1;X3]1[CH2][CH2][CH2][CH2][CH2]1', "[CH1]1CCCCC1", 0.3),
    Substitution(['Fmoc'], 'C(=O)O[CH2][CH1]1c([cH1][cH1][cH1][cH1]2)c2c3c1[cH1][cH1][cH1][cH1]3',
                 "[C](=O)OCC1c(cccc2)c2c3c1cccc3", 0.6),
    Substitution(['Mes'], '[cH0]1c([CH3])cc([CH3])cc([CH3])1', "[c]1c(C)cc(C)cc(C)1", 0.5),
    Substitution(['OMs'], '[OH0;D2]S(=O)(=O)[CH3]', "[O]S(=O)(=O)C", 0.7),
    Substitution(['Ms'], 'S(=O)(=O)[CH3]', "[S](=O)(=O)C", 0.2),
    Substitution(['Ph'], '[cH0]1[cH][cH][cH1][cH][cH]1', "[c]1ccccc1", 0.5),
    Substitution(['PMB'], '[CH2;D2][cH0]1[cH1][cH1][cH0](O[CH3])[cH1][cH1]1', "[CH2]c1ccc(OC)cc1", 0.2),
    Substitution(['Py'], '[cH0]1[n;+0][cH1][cH1][cH1][cH1]1', "[c]1ncccc1", 0.1),
    Substitution(['SEM'], '[CH2;D2][CH2][Si]([CH3])([CH3])[CH3]', "[CH2]CSi(C)(C)C", 0.2),
    Substitution(['Suc'], 'C(=O)[CH2][CH2]C(=O)[OH]', "[C](=O)CCC(=O)O", 0.2),
    Substitution(['TBS'], '[Si]([CH3])([CH3])C([CH3])([CH3])[CH3]', "[Si](C)(C)C(C)(C)C", 0.5),
    Substitution(['TBZ'], 'C(=S)[cH]1[cH][cH][cH1][cH][cH]1', "[C](=S)c1ccccc1", 0.2),
    Substitution(['OTf'], '[OH0;D2]S(=O)(=O)C(F)(F)F', "[O]S(=O)(=O)C(F)(F)F", 0.7),
    Substitution(['Tf'], 'S(=O)(=O)C(F)(F)F', "[S](=O)(=O)C(F)(F)F", 0.2),
    Substitution(['TFA'], 'C(=O)C(F)(F)F', "[C](=O)C(F)(F)F", 0.3),
    Substitution(['TMS'], '[Si]([CH3])([CH3])[CH3]', "[Si](C)(C)C", 0.5),
    Substitution(['Ts'], 'S(=O)(=O)c1[cH1][cH1][cH0]([CH3])[cH1][cH1]1', "[S](=O)(=O)c1ccc(C)cc1", 0.6),  
    Substitution(['OMe', 'MeO'], '[OH0;D2][CH3;D1]', "[O]C", 0.3),
    Substitution(['SMe', 'MeS'], '[SH0;D2][CH3;D1]', "[S]C", 0.3),
    Substitution(['NMe', 'MeN'], '[N;X3][CH3;D1]', "[NH]C", 0.3),
    Substitution(['Me'], '[CH3;D1]', "[CH3]", 0.1),
    Substitution(['OEt', 'EtO'], '[OH0;D2][CH2;D2][CH3]', "[O]CC", 0.5),
    Substitution(['Et', 'C2H5'], '[CH2;D2][CH3]', "[CH2]C", 0.3),
    Substitution(['Pr', 'nPr', 'n-Pr'], '[CH2;D2][CH2;D2][CH3]', "[CH2]CC", 0.3),
    Substitution(['Bu', 'nBu', 'n-Bu'], '[CH2;D2][CH2;D2][CH2;D2][CH3]', "[CH2]CCC", 0.3),
    Substitution(['iPr', 'i-Pr'], '[CH1;D3]([CH3])[CH3]', "[CH1](C)C", 0.2),
    Substitution(['iBu', 'i-Bu'], '[CH2;D2][CH1;D3]([CH3])[CH3]', "[CH2]C(C)C", 0.2),
    Substitution(['OiBu'], '[OH0;D2][CH2;D2][CH1;D3]([CH3])[CH3]', "[O]CC(C)C", 0.2),
    Substitution(['OtBu'], '[OH0;D2][CH0]([CH3])([CH3])[CH3]', "[O]C(C)(C)C", 0.6),
    Substitution(['tBu', 't-Bu'], '[CH0]([CH3])([CH3])[CH3]', "[C](C)(C)C", 0.3),
    Substitution(['CF3', 'F3C'], '[CH0;D4](F)(F)F', "[C](F)(F)F", 0.5),
    Substitution(['NCF3', 'F3CN'], '[N;X3][CH0;D4](F)(F)F', "[NH]C(F)(F)F", 0.5),
    Substitution(['OCF3', 'F3CO'], '[OH0;X2][CH0;D4](F)(F)F', "[O]C(F)(F)F", 0.5),
    Substitution(['CCl3'], '[CH0;D4](Cl)(Cl)Cl', "[C](Cl)(Cl)Cl", 0.5),
    Substitution(['CO2H', 'HO2C', 'COOH'], 'C(=O)[OH]', "[C](=O)O", 0.5),  
    Substitution(['CN', 'NC'], 'C#[ND1]', "[C]#N", 0.5),
    Substitution(['OCH3', 'H3CO'], '[OH0;D2][CH3]', "[O]C", 0.4),
    Substitution(['SO3H'], 'S(=O)(=O)[OH]', "[S](=O)(=O)O", 0.4),
]
ABBREVIATIONS = {abbrv: sub for sub in SUBSTITUTIONS for abbrv in sub.abbrvs}
charge_labels = [18,19,20,21,22]
def _expand_abbreviation(abbrev):
    if abbrev in ABBREVIATIONS:
        return ABBREVIATIONS[abbrev].smiles
    if abbrev in RGROUP_SYMBOLS or (abbrev[0] == 'R' and abbrev[1:].isdigit()):
        if abbrev[1:].isdigit():
            return f'[{abbrev[1:]}*]'
        return '*'
    return f'[{abbrev}]'
def Val_extract_atom_info(error_message):
    pattern = r"Explicit valence for atom # (\d+) (\w), (\d+)"
    pattern2 =r"Explicit valence for atom # (\d+) (\w) "
    if not isinstance(error_message, type('strs')):
        error_message=str(error_message)
    match = re.search(pattern, error_message)
    match2 = re.search(pattern2, error_message)
    if match:
        atomid = int(match.group(1))  
        atomType = match.group(2)     
        valence = int(match.group(3)) 
        return atomid, atomType, valence
    elif match2:
        atomid = int(match2.group(1))  
        atomType = match2.group(2)     
        return atomid, atomType, None
    else:
        raise ValueError("无法从错误信息中提取原子信息")
def calculate_charge_adjustment(atom_symbol, current_valence):
    if atom_symbol not in VALENCES:
        raise ValueError(f"未知的原子符号: {atom_symbol}")
    max_valence = max(VALENCES[atom_symbol])
    if current_valence is None:
        current_valence=max_valence
    if current_valence > max_valence:
        charge_adjustment = current_valence - max_valence
        return charge_adjustment 
    else:
        return 0
from rdkit.Chem import rdchem, RWMol, CombineMols
def expandABB(mol,ABBREVIATIONS, placeholder_atoms):
    mols = [mol]
    for idx in sorted(placeholder_atoms.keys(), reverse=True):
        group = placeholder_atoms[idx]  
        submol = Chem.MolFromSmiles(ABBREVIATIONS[group].smiles)  
        submol_rw = RWMol(submol)  
        anchor_atom_idx = 0  
        new_mol = RWMol(mol)
        placeholder_idx = idx
        neighbors = [nb.GetIdx() for nb in new_mol.GetAtomWithIdx(placeholder_idx).GetNeighbors()]
        bonds_to_remove = []  
        for bond in new_mol.GetBonds():
            if bond.GetBeginAtomIdx() == placeholder_idx or bond.GetEndAtomIdx() == placeholder_idx:
                bonds_to_remove.append((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()))
        for bond in bonds_to_remove:
            new_mol.RemoveBond(bond[0], bond[1])
        new_mol.RemoveAtom(placeholder_idx)
        new_neighbors = []
        for neighbor in neighbors:
            if neighbor < placeholder_idx:
                new_neighbors.append(neighbor)
            else:
                new_neighbors.append(neighbor - 1)  
        new_mol = RWMol(CombineMols(new_mol, submol_rw))
        new_anchor_idx = new_mol.GetNumAtoms() - len(submol_rw.GetAtoms()) + anchor_atom_idx
        for neighbor in new_neighbors:
            new_mol.AddBond(neighbor, new_anchor_idx, Chem.BondType.SINGLE)
            a1=new_mol.GetAtomWithIdx(neighbor)
            a2=new_mol.GetAtomWithIdx(new_anchor_idx)
            a1.SetNumRadicalElectrons(0)
            a2.SetNumRadicalElectrons(0)
        mol = new_mol
        mols.append(mol)
    Chem.SanitizeMol(mols[-1])
    modified_smiles = Chem.MolToSmiles(mols[-1])
    return mols[-1], modified_smiles
def output_to_smiles(output,idx_to_labels,bond_labels,result):
    x_center = (output["boxes"][:, 0] + output["boxes"][:, 2]) / 2
    y_center = (output["boxes"][:, 1] + output["boxes"][:, 3]) / 2
    center_coords = torch.stack((x_center, y_center), dim=1)
    output = {'bbox':         output["boxes"].to("cpu").numpy(),
                'bbox_centers': center_coords.to("cpu").numpy(),
                'scores':       output["scores"].to("cpu").numpy(),
                'pred_classes': output["labels"].to("cpu").numpy()}
    atoms_list, bonds_list,charge = bbox_to_graph_with_charge(output,
                                                idx_to_labels=idx_to_labels,
                                                bond_labels=bond_labels,
                                                result=result)
    smiles, mol= mol_from_graph_with_chiral(atoms_list, bonds_list,charge)
    abc=[atoms_list, bonds_list,charge ]
    if isinstance(smiles, type(None)):
        print(f"get atoms_list problems")
    elif isinstance(atoms_list,type(None)):
        print(f"get atoms_list problems")
    return abc,smiles,mol,output
def output_to_smiles2(output,idx_to_labels,bond_labels,result):
    x_center = (output["boxes"][:, 0] + output["boxes"][:, 2]) / 2
    y_center = (output["boxes"][:, 1] + output["boxes"][:, 3]) / 2
    center_coords = torch.stack((x_center, y_center), dim=1)
    output = {'bbox':         output["boxes"].to("cpu").numpy(),
                'bbox_centers': center_coords.to("cpu").numpy(),
                'scores':       output["scores"].to("cpu").numpy(),
                'pred_classes': output["labels"].to("cpu").numpy()}
    atoms_list, bonds_list,charge = bbox_to_graph_with_charge(output,
                                                idx_to_labels=idx_to_labels,
                                                bond_labels=bond_labels,
                                                result=result)
    smiles, mol= mol_from_graph_with_chiral(atoms_list, bonds_list,charge)
    abc=[atoms_list, bonds_list,charge ]
    if isinstance(smiles, type(None)):
        print(f"get atoms_list problems")
    elif isinstance(atoms_list,type(None)):
        print(f"get atoms_list problems")
    return abc,smiles,mol,output
def bbox_to_graph(output, idx_to_labels, bond_labels,result):
    atoms_mask = np.array([True if ins not in bond_labels else False for ins in output['pred_classes']])
    atoms_list = [idx_to_labels[a] for a in output['pred_classes'][atoms_mask]]
    atoms_list = pd.DataFrame({'atom': atoms_list,
                            'x':    output['bbox_centers'][atoms_mask, 0],
                            'y':    output['bbox_centers'][atoms_mask, 1]})
    for idx, row in atoms_list.iterrows():
        if row.atom[-1] != '0':
            if row.atom[-2] != '-':
                overlapping = atoms_list[atoms_list.atom.str.startswith(row.atom[:-1])]
            else:
                overlapping = atoms_list[atoms_list.atom.str.startswith(row.atom[:-2])]
            kdt = cKDTree(overlapping[['x', 'y']])
            dists, neighbours = kdt.query([row.x, row.y], k=2)
            if dists[1] < 7:
                atoms_list.drop(overlapping.index[neighbours[1]], axis=0, inplace=True)
    bonds_list = []
    for bbox, bond_type, score in zip(output['bbox'][np.logical_not(atoms_mask)],
                                    output['pred_classes'][np.logical_not(atoms_mask)],
                                    output['scores'][np.logical_not(atoms_mask)]):
        if idx_to_labels[bond_type] in ['-','SINGLE', 'NONE', 'ENDUPRIGHT', 'BEGINWEDGE', 'BEGINDASH', 'ENDDOWNRIGHT']:
            _margin = 5
        else:
            _margin = 8
        anchor_positions = (bbox + [_margin, _margin, -_margin, -_margin]).reshape([2, -1])
        oposite_anchor_positions = anchor_positions.copy()
        oposite_anchor_positions[:, 1] = oposite_anchor_positions[:, 1][::-1]
        anchor_positions = np.concatenate([anchor_positions, oposite_anchor_positions])
        atoms_pos = atoms_list[['x', 'y']].values
        kdt = cKDTree(atoms_pos)
        dists, neighbours = kdt.query(anchor_positions, k=1)
        if np.argmin((dists[0] + dists[1], dists[2] + dists[3])) == 0:
            begin_idx, end_idx = neighbours[:2]
        else:
            begin_idx, end_idx = neighbours[2:]
        if begin_idx != end_idx:
            bonds_list.append((begin_idx, end_idx, idx_to_labels[bond_type], idx_to_labels[bond_type], score))
        else:
            continue
    return atoms_list, bonds_list
def calculate_distance(coord1, coord2):
    return math.sqrt((coord1[0] - coord2[0])**2 + (coord1[1] - coord2[1])**2)
def assemble_atoms_with_charges(atom_list, charge_list):
    used_charge_indices=set()
    atom_list = atom_list.reset_index(drop=True)
    kdt = cKDTree(atom_list[['x','y']])
    for i, charge in charge_list.iterrows():
        if i in used_charge_indices:
            continue
        charge_=charge['charge']
        dist, idx_atom=kdt.query([charge_list.x[i],charge_list.y[i]], k=1)
        if idx_atom not in atom_list.index:
            print(f"Warning: idx_atom {idx_atom} is out of range for atom_list.")
            continue  
        atom_str = atom_list.iloc[idx_atom]['atom']
        if atom_str=='*':
            atom_=atom_str + charge_
        else:
            try:
                atom_ = re.findall(r'[A-Za-z*]+', atom_str)[0] + charge_
            except Exception as e:
                print(atom_str,charge_,charge_list)
                print(f"@assemble_atoms_with_charges\n {e}\n{atom_list}")
                atom_=atom_str + charge_
        atom_list.loc[idx_atom,'atom']=atom_
    return atom_list
def assemble_atoms_with_charges2(atom_list, charge_list, max_distance=10):
    used_charge_indices = set()
    for idx, atom in atom_list.iterrows():
        atom_coord = atom['x'],atom['y']
        atom_label = atom['atom']
        closest_charge = None
        min_distance = float('inf')
        for i, charge in charge_list.iterrows():
            if i in used_charge_indices:
                continue
            charge_coord = charge['x'],charge['y']
            charge_label = charge['charge']
            distance = calculate_distance(atom_coord, charge_coord)
            if distance <= max_distance and distance < min_distance:
                closest_charge = charge
                min_distance = distance
        if closest_charge is not None:
            if closest_charge['charge'] == '1':
                charge_ = '+'
            else:
                charge_ = closest_charge['charge']
            atom_ = atom['atom'] + charge_
            atom_list.loc[idx,'atom'] = atom_
            used_charge_indices.add(tuple(charge))
        else:
            atom_list.loc[idx,'atom'] = atom['atom'] + '0'
    return atom_list
def calculate_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area
    iou = inter_area / union_area if union_area > 0 else 0
    return iou
def bbox_to_graph_with_charge(output, idx_to_labels, bond_labels,result):
    bond_labels_pre=bond_labels
    atoms_mask = np.array([True if ins not in bond_labels and ins not in charge_labels else False for ins in output['pred_classes']])
    try:
        atoms_list = [idx_to_labels[a] for a in output['pred_classes'][atoms_mask]]
        if isinstance(atoms_list, pd.Series) and atoms_list.empty:
            return None, None, None
        else:
            atoms_list = pd.DataFrame({'atom': atoms_list,
                                    'x':    output['bbox_centers'][atoms_mask, 0],
                                    'y':    output['bbox_centers'][atoms_mask, 1],
                                    'bbox':  output['bbox'][atoms_mask].tolist() ,
                                    'scores': output['scores'][atoms_mask].tolist(),
                                    })
    except Exception as e:
        print(output['pred_classes'][atoms_mask].dtype,output['pred_classes'][atoms_mask])
        print(e)
        print(idx_to_labels)
    charge_mask = np.array([True if ins in charge_labels else False for ins in output['pred_classes']])
    charge_list = [idx_to_labels[a] for a in output['pred_classes'][charge_mask]]
    charge_list = pd.DataFrame({'charge': charge_list,
                        'x':    output['bbox_centers'][charge_mask, 0],
                        'y':    output['bbox_centers'][charge_mask, 1],
                        'scores':    output['scores'][charge_mask],
                        })
    try:
        atoms_list['atom'] = atoms_list['atom']+'0'
    except Exception as e:
        print(e)
        print(atoms_list['atom'],'atoms_list["atom"] @@ adding 0 ')
    if len(charge_list) > 0:
        atoms_list = assemble_atoms_with_charges(atoms_list,charge_list)
    for idx, row in atoms_list.iterrows():
        if row.atom[-1] != '0':
            try:
                if row.atom[-2] != '-':
                    overlapping = atoms_list[atoms_list.atom.str.startswith(row.atom[:-1])]
            except Exception as e:
                print(row.atom,"@rin case atoms with sign gets detected two times")
                print(e)
            else:
                overlapping = atoms_list[atoms_list.atom.str.startswith(row.atom[:-2])]
            kdt = cKDTree(overlapping[['x', 'y']])
            dists, neighbours = kdt.query([row.x, row.y], k=2)
            if dists[1] < 7:
                atoms_list.drop(overlapping.index[neighbours[1]], axis=0, inplace=True)
    bonds_list = []
    bond_mask=np.logical_not(atoms_mask) & np.logical_not(charge_mask)
    for bbox, bond_type, score in zip(output['bbox'][bond_mask],  
                                    output['pred_classes'][bond_mask],
                                    output['scores'][bond_mask]):
        if len(idx_to_labels)==23:
            if idx_to_labels[bond_type] in ['-','SINGLE', 'NONE', 'ENDUPRIGHT', 'BEGINWEDGE', 'BEGINDASH', 'ENDDOWNRIGHT']:
                _margin = 5
            else:
                _margin = 8
        elif len(idx_to_labels)==30:
            _margin=0
        elif len(idx_to_labels)==24:
            _margin=0
        anchor_positions = (bbox + [_margin, _margin, -_margin, -_margin]).reshape([2, -1])
        oposite_anchor_positions = anchor_positions.copy()
        oposite_anchor_positions[:, 1] = oposite_anchor_positions[:, 1][::-1]
        anchor_positions = np.concatenate([anchor_positions, oposite_anchor_positions])
        atoms_pos = atoms_list[['x', 'y']].values
        kdt = cKDTree(atoms_pos)
        dists, neighbours = kdt.query(anchor_positions, k=1)
        if np.argmin((dists[0] + dists[1], dists[2] + dists[3])) == 0:
            begin_idx, end_idx = neighbours[:2]
        else:
            begin_idx, end_idx = neighbours[2:]
        if begin_idx != end_idx: 
            if bond_type in bond_labels:
                bonds_list.append((begin_idx, end_idx, idx_to_labels[bond_type], idx_to_labels[bond_type], score))
            else:
                print(f'this box may be charges box not bonds {[bbox, bond_type, score ]}')
        else:
            continue
    return atoms_list, bonds_list,charge_list
def parse_atom(node):
    s10 = [str(x) for x in range(10)]
    if 'other' in node:
        a = '*'
        if '-' in node or '+' in node:
            fc = -1 if node[-1] == '-' else 1
        else:
            fc = int(node[-2:]) if node[-2:] in s10 else 0
    elif node[-1] in s10:
        if '-' in node or '+' in node:
            fc = -1 if node[-1] == '-' else 1
            a = node[:-1]
        else:
            a = node[:-1]
            fc = int(node[-1])
    elif node[-1] == '+':
        a = node[:-1]
        fc = 1
    elif node[-1] == '-':
        a = node[:-1]
        fc = -1
    else:
        a = node
        fc = 0
    return a, fc
def mol_from_graph_with_chiral(atoms_list, bonds,charge):
    mol = RWMol()
    nodes_idx = {}
    atoms = atoms_list.atom.values.tolist()
    coords = [(row['x'], 300-row['y'], 0) for index, row in atoms_list.iterrows()]
    coords = tuple(coords)
    coords = tuple(tuple(num / 100 for num in sub_tuple) for sub_tuple in coords)
    for i in range(len(bonds)):
        idx_1, idx_2, bond_type, bond_dir, score = bonds[i]
        if bond_type in ['-', 'NONE', 'ENDUPRIGHT', 'BEGINWEDGE', 'BEGINDASH', 'ENDDOWNRIGHT']:
            bonds[i] = (idx_1, idx_2, 'SINGLE', bond_dir, score)
        elif bond_type == '=':
            bonds[i] = (idx_1, idx_2, 'DOUBLE', bond_dir, score)
        elif bond_type == '#':
            bonds[i] = (idx_1, idx_2, 'TRIPLE', bond_dir, score)
    bond_types = {'SINGLE':   Chem.rdchem.BondType.SINGLE,
                'DOUBLE':   Chem.rdchem.BondType.DOUBLE,
                'TRIPLE':   Chem.rdchem.BondType.TRIPLE,
                'AROMATIC': Chem.rdchem.BondType.AROMATIC,
                'single':   Chem.rdchem.BondType.SINGLE,
                '=':        Chem.rdchem.BondType.DOUBLE,
                '#':        Chem.rdchem.BondType.TRIPLE,
                ':':        Chem.rdchem.BondType.AROMATIC}
    bond_dirs = {'NONE':    Chem.rdchem.BondDir.NONE,
            'ENDUPRIGHT':   Chem.rdchem.BondDir.ENDUPRIGHT,
            'BEGINWEDGE':   Chem.rdchem.BondDir.BEGINWEDGE,
            'BEGINDASH':    Chem.rdchem.BondDir.BEGINDASH,
            'ENDDOWNRIGHT': Chem.rdchem.BondDir.ENDDOWNRIGHT,}
    debug=True
    if debug:
        placeholder_atoms = {}
        s10=[str(x) for x in range(10)]
        for idx, node in enumerate(atoms):
            if 'other' in node:
                a='*'
                if '-' in node or '+' in node:
                    if node[-1] =='-':
                        fc = -1
                    elif [-1] =='+':
                        fc = 1
                    else:      
                        fc = int(node[-2:])
                else:
                    fc = int(node[-1])
            elif node[-1] in s10:
                if '-' in node or '+' in node:
                    if node[-1] =='-':
                        fc = -1
                        a = node[:-1]
                    elif [-1] =='+':
                        fc = 1
                        a = node[:-1]
                    else:      
                        fc = int(node[-2:])
                        a = node[:-2]
                else:
                    a = node[:-1]
                    fc = int(node[-1])
            elif node[-1]=='+':
                a = node[:-1]
                fc = 1
            elif  node[-1]=='-':
                a = node[:-1]
                fc = -1
            else:
                a = node
                fc = 0
            if a in ['H', 'C', 'O', 'N', 'Cl', 'Br', 'S', 'F', 'B', 'I', 'P', 'Si']:
                ad = Chem.Atom(a)
            elif a in ABBREVIATIONS:
                smi = ABBREVIATIONS[a].smiles
                ad = Chem.Atom("*")
                placeholder_atoms[idx] = a 
            else:
                ad = Chem.Atom("*")
            ad.SetFormalCharge(fc)
            atom_idx = mol.AddAtom(ad)
            nodes_idx[idx] = atom_idx
        existing_bonds = set()
        for idx_1, idx_2, bond_type, bond_dir, score in bonds:
            if (idx_1 in nodes_idx) and (idx_2 in nodes_idx):
                if (idx_1, idx_2) not in existing_bonds and (idx_2, idx_1) not in existing_bonds:
                    try:
                        mol.AddBond(nodes_idx[idx_1], nodes_idx[idx_2], bond_types[bond_type])
                    except Exception as e:
                        print([idx_1, idx_2, bond_type, bond_dir, score],f"erro @add bonds ")
                        print(f"erro@add existing_bonds: {e}\n{bonds}")
                        continue
            existing_bonds.add((idx_1, idx_2))
            if Chem.MolFromSmiles(Chem.MolToSmiles(mol.GetMol())):
                prev_mol = copy.deepcopy(mol)
            else:
                mol = copy.deepcopy(prev_mol)
        chiral_centers = Chem.FindMolChiralCenters(
            mol, includeUnassigned=True, includeCIP=False, useLegacyImplementation=False)
        chiral_center_ids = [idx for idx, _ in chiral_centers] 
        for id in chiral_center_ids:
            for index, tup in enumerate(bonds):
                if id == tup[1]:
                    new_tup = tuple([tup[1], tup[0], tup[2], tup[3], tup[4]])
                    bonds[index] = new_tup
                    mol.RemoveBond(int(tup[0]), int(tup[1]))
                    try:
                        mol.AddBond(int(tup[1]), int(tup[0]), bond_types[tup[2]])
                    except Exception as e:
                        print( index, tup, id)
                        print(f"bonds: {bonds}")
                        print(f"erro@chiral_center_ids: {e}")
        mol = mol.GetMol()
        mol.RemoveAllConformers()
        conf = Chem.Conformer(mol.GetNumAtoms())
        conf.Set3D(True)
        for i, (x, y, z) in enumerate(coords):
            conf.SetAtomPosition(i, (x, y, z))
        mol.AddConformer(conf)
        Chem.AssignStereochemistryFrom3D(mol)
        bonds_ = [[row[0], row[1], row[3]] for row in bonds]
        n_atoms=len(atoms)
        for i in chiral_center_ids:
            for j in range(n_atoms):
                b_=mol.GetBondBetweenAtoms(i, j)
                if [i,j,'BEGINWEDGE'] in bonds_ and b_:
                    b_.SetBondDir(bond_dirs['BEGINWEDGE'])
                elif [i,j,'BEGINDASH'] in bonds_ and b_:
                    b_.SetBondDir(bond_dirs['BEGINDASH'])
        try:
            Chem.SanitizeMol(mol)
        except Exception as e:
            problems = Chem.DetectChemistryProblems(mol)
            if len(problems)>0:
                print(F"get problems",len(problems))
                for p in problems:
                    atomid, atomType, valence=Val_extract_atom_info(p.Message())
                    charge_adjustment=calculate_charge_adjustment(atomType, valence)
                    mol.GetAtomWithIdx(atomid).SetFormalCharge(charge_adjustment)
        Chem.DetectBondStereochemistry(mol)
        Chem.AssignChiralTypesFromBondDirs(mol)
        Chem.AssignStereochemistry(mol)
        smiles=Chem.MolToSmiles(mol)
        return smiles,mol
    else:
        try:
            placeholder_atoms = {}
            s10=[str(x) for x in range(10)]
            for idx, node in enumerate(atoms):
                if 'other' in node:
                    a='*'
                    if '-' in node or '+' in node:
                        if node[-1] =='-':
                            fc = -1
                        elif [-1] =='+':
                            fc = 1
                        else:      
                            fc = int(node[-2:])
                    else:
                        fc = int(node[-1])
                elif node[-1] in s10:
                    if '-' in node or '+' in node:
                        if node[-1] =='-':
                            fc = -1
                            a = node[:-1]
                        elif [-1] =='+':
                            fc = 1
                            a = node[:-1]
                        else:      
                            fc = int(node[-2:])
                            a = node[:-2]
                    else:
                        a = node[:-1]
                        fc = int(node[-1])
                elif node[-1]=='+':
                    a = node[:-1]
                    fc = 1
                elif  node[-1]=='-':
                    a = node[:-1]
                    fc = -1
                else:
                    a = node
                    fc = 0
                if a in ['H', 'C', 'O', 'N', 'Cl', 'Br', 'S', 'F', 'B', 'I', 'P', 'Si']:
                    ad = Chem.Atom(a)
                elif a in ABBREVIATIONS:
                    smi = ABBREVIATIONS[a].smiles
                    submol = Chem.MolFromSmiles(smi)
                    ad = Chem.Atom("*")
                    placeholder_atoms[idx] = a 
                else:
                    ad = Chem.Atom("*")
                ad.SetFormalCharge(fc)
                atom_idx = mol.AddAtom(ad)
                nodes_idx[idx] = atom_idx
            existing_bonds = set()
            for idx_1, idx_2, bond_type, bond_dir, score in bonds:
                if (idx_1 in nodes_idx) and (idx_2 in nodes_idx):
                    if (idx_1, idx_2) not in existing_bonds and (idx_2, idx_1) not in existing_bonds:
                        try:
                            mol.AddBond(nodes_idx[idx_1], nodes_idx[idx_2], bond_types[bond_type])
                        except Exception as e:
                            print([idx_1, idx_2, bond_type, bond_dir, score],f"erro @add bonds ")
                            print(f"erro@add existing_bonds: {e}\n{bonds}")
                            continue
                existing_bonds.add((idx_1, idx_2))
                if Chem.MolFromSmiles(Chem.MolToSmiles(mol.GetMol())):
                    prev_mol = copy.deepcopy(mol)
                else:
                    mol = copy.deepcopy(prev_mol)
            chiral_centers = Chem.FindMolChiralCenters(
                mol, includeUnassigned=True, includeCIP=False, useLegacyImplementation=False)
            chiral_center_ids = [idx for idx, _ in chiral_centers] 
            for id in chiral_center_ids:
                for index, tup in enumerate(bonds):
                    if id == tup[1]:
                        new_tup = tuple([tup[1], tup[0], tup[2], tup[3], tup[4]])
                        bonds[index] = new_tup
                        mol.RemoveBond(int(tup[0]), int(tup[1]))
                        try:
                            mol.AddBond(int(tup[1]), int(tup[0]), bond_types[tup[2]])
                        except Exception as e:
                            print( index, tup, id)
                            print(f"bonds: {bonds}")
                            print(f"erro@chiral_center_ids: {e}")
            mol = mol.GetMol()
            mol.RemoveAllConformers()
            conf = Chem.Conformer(mol.GetNumAtoms())
            conf.Set3D(True)
            for i, (x, y, z) in enumerate(coords):
                conf.SetAtomPosition(i, (x, y, z))
            mol.AddConformer(conf)
            Chem.AssignStereochemistryFrom3D(mol)
            bonds_ = [[row[0], row[1], row[3]] for row in bonds]
            n_atoms=len(atoms)
            for i in chiral_center_ids:
                for j in range(n_atoms):
                    if [i,j,'BEGINWEDGE'] in bonds_:
                        mol.GetBondBetweenAtoms(i, j).SetBondDir(bond_dirs['BEGINWEDGE'])
                    elif [i,j,'BEGINDASH'] in bonds_:
                        mol.GetBondBetweenAtoms(i, j).SetBondDir(bond_dirs['BEGINDASH'])
            Chem.SanitizeMol(mol)
            Chem.DetectBondStereochemistry(mol)
            Chem.AssignChiralTypesFromBondDirs(mol)
            Chem.AssignStereochemistry(mol)
            smiles=Chem.MolToSmiles(mol)
            return smiles,mol
        except Chem.rdchem.AtomValenceException as e:
            print(f"捕获到 AtomValenceException 异常@@\n{e}")
            print(atoms,f"idx:{idx},atoms[idx]::{atoms[idx]}")
            return None,mol
        except Exception as e:
            print(f"捕获到   异常@@{e}")
            print(f"Error@@node {node} atom@@ {a} \n")
            print(atoms,f"idx:{idx},atoms[idx]::{atoms[idx]}")
            print("bonds:",bonds)
            print("charge:",charge)
            return None,mol
def mol_from_graph_without_chiral(atoms, bonds):
    mol = RWMol()
    nodes_idx = {}
    for i in range(len(bonds)):
        idx_1, idx_2, bond_type, bond_dir, score = bonds[i]
        if bond_type in  ['-', 'NONE', 'ENDUPRIGHT', 'BEGINWEDGE', 'BEGINDASH', 'ENDDOWNRIGHT']:
            bonds[i] = (idx_1, idx_2, 'SINGLE', bond_dir, score)
        elif bond_type == '=':
            bonds[i] = (idx_1, idx_2, 'DOUBLE', bond_dir, score)
        elif bond_type == '#':
            bonds[i] = (idx_1, idx_2, 'TRIPLE', bond_dir, score)
    bond_types = {'SINGLE':   Chem.rdchem.BondType.SINGLE,
                'DOUBLE':   Chem.rdchem.BondType.DOUBLE,
                'TRIPLE':   Chem.rdchem.BondType.TRIPLE,
                'AROMATIC': Chem.rdchem.BondType.AROMATIC,
                'single':   Chem.rdchem.BondType.SINGLE,
                '=':        Chem.rdchem.BondType.DOUBLE,
                '#':        Chem.rdchem.BondType.TRIPLE,
                ':':        Chem.rdchem.BondType.AROMATIC}
    try:
        for idx, node in enumerate(atoms):
            if ('0' in node) or ('1' in node):
                a = node[:-1]
                fc = int(node[-1])
            if '-1' in node:
                a = node[:-2]
                fc = -1
            a = Chem.Atom(a)
            a.SetFormalCharge(fc)
            atom_idx = mol.AddAtom(a)
            nodes_idx[idx] = atom_idx
        existing_bonds = set()
        for idx_1, idx_2, bond_type, bond_dir, score in bonds:
            if (idx_1 in nodes_idx) and (idx_2 in nodes_idx):
                if (idx_1, idx_2) not in existing_bonds and (idx_2, idx_1) not in existing_bonds:
                    try:
                        mol.AddBond(nodes_idx[idx_1], nodes_idx[idx_2], bond_types[bond_type])
                    except:
                        continue
            existing_bonds.add((idx_1, idx_2))
            if Chem.MolFromSmiles(Chem.MolToSmiles(mol.GetMol())):
                prev_mol = copy.deepcopy(mol)
            else:
                mol = copy.deepcopy(prev_mol)
        mol = mol.GetMol()
        mol = Chem.MolFromSmiles(Chem.MolToSmiles(mol))
        return Chem.MolToSmiles(mol)
    except Chem.rdchem.AtomValenceException as e:
        print("捕获到 AtomValenceException 异常")