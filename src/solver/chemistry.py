import copy
import traceback
import numpy as np
import multiprocessing
import itertools
import rdkit
import rdkit.Chem as Chem
rdkit.RDLogger.DisableLog('rdApp.*')
def is_valid_mol(s, format_='atomtok'):
    if format_ == 'atomtok':
        mol = Chem.MolFromSmiles(s)
    elif format_ == 'inchi':
        if not s.startswith('InChI=1S'):
            s = f"InChI=1S/{s}"
        mol = Chem.MolFromInchi(s)
    else:
        raise NotImplemented
    return mol is not None
def _convert_smiles_to_inchi(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        inchi = Chem.MolToInchi(mol)
    except:
        inchi = None
    return inchi
def convert_smiles_to_inchi(smiles_list, num_workers=16):
    with multiprocessing.Pool(num_workers) as p:
        inchi_list = p.map(_convert_smiles_to_inchi, smiles_list, chunksize=128)
    n_success = sum([x is not None for x in inchi_list])
    r_success = n_success / len(inchi_list)
    inchi_list = [x if x else 'InChI=1S/H2O/h1H2' for x in inchi_list]
    return inchi_list, r_success
def merge_inchi(inchi1, inchi2):
    replaced = 0
    inchi1 = copy.deepcopy(inchi1)
    for i in range(len(inchi1)):
        if inchi1[i] == 'InChI=1S/H2O/h1H2':
            inchi1[i] = inchi2[i]
            replaced += 1
    return inchi1, replaced
def _get_num_atoms(smiles):
    try:
        return Chem.MolFromSmiles(smiles).GetNumAtoms()
    except:
        return 0
def get_num_atoms(smiles, num_workers=16):
    if type(smiles) is str:
        return _get_num_atoms(smiles)
    with multiprocessing.Pool(num_workers) as p:
        num_atoms = p.map(_get_num_atoms, smiles)
    return num_atoms
def normalize_nodes(nodes, flip_y=True):
    x, y = nodes[:, 0], nodes[:, 1]
    minx, maxx = min(x), max(x)
    miny, maxy = min(y), max(y)
    x = (x - minx) / max(maxx - minx, 1e-6)
    if flip_y:
        y = (maxy - y) / max(maxy - miny, 1e-6)
    else:
        y = (y - miny) / max(maxy - miny, 1e-6)
    return np.stack([x, y], axis=1)
def _verify_chirality(mol, coords, bonds):
    try:
        n = mol.GetNumAtoms()
        mol_tmp = mol.GetMol()
        Chem.SanitizeMol(mol_tmp)
        chiral_centers = Chem.FindMolChiralCenters(
            mol_tmp, includeUnassigned=True, includeCIP=False, useLegacyImplementation=False)
        chiral_center_ids = [idx for idx, _ in chiral_centers]  
        for bond in mol.GetBonds():
            if bond.GetBondType() == Chem.BondType.SINGLE:
                bond.SetBondDir(Chem.BondDir.NONE)
        conf = Chem.Conformer(n)
        conf.Set3D(True)
        for i, (x, y) in enumerate(coords):
            conf.SetAtomPosition(i, (x, 1 - y, 0))
        mol.AddConformer(conf)
        Chem.SanitizeMol(mol)
        Chem.AssignStereochemistryFrom3D(mol)
        mol.RemoveAllConformers()
        conf = Chem.Conformer(n)
        conf.Set3D(False)
        for i, (x, y) in enumerate(coords):
            conf.SetAtomPosition(i, (x, 1 - y, 0))
        mol.AddConformer(conf)
        Chem.SanitizeMol(mol)
        Chem.AssignChiralTypesFromBondDirs(mol)
        Chem.AssignStereochemistry(mol, force=True)
        bond_dirs = {'NONE':    Chem.rdchem.BondDir.NONE,
            'ENDUPRIGHT':   Chem.rdchem.BondDir.ENDUPRIGHT,
            'BEGINWEDGE':   Chem.rdchem.BondDir.BEGINWEDGE,
            'BEGINDASH':    Chem.rdchem.BondDir.BEGINDASH,
            'ENDDOWNRIGHT': Chem.rdchem.BondDir.ENDDOWNRIGHT,}
        bonds_ = [[row[0], row[1], row[3]] for row in bonds]
        for i in chiral_center_ids:
            for j in range(n):
                if [i,j,'ENDUPRIGHT'] in bonds_ or [j,i,'ENDUPRIGHT'] in bonds_:
                    mol.RemoveBond(i, j)
                    mol.AddBond(i, j, Chem.BondType.SINGLE)
                    mol.GetBondBetweenAtoms(i, j).SetBondDir(bond_dirs['ENDUPRIGHT'])
                elif [i,j,'BEGINWEDGE'] in bonds_ or [j,i,'BEGINWEDGE'] in bonds_:
                    mol.RemoveBond(i, j)
                    mol.AddBond(i, j, Chem.BondType.SINGLE)
                    mol.GetBondBetweenAtoms(i, j).SetBondDir(bond_dirs['BEGINWEDGE'])
                elif [i,j,'BEGINDASH'] in bonds_ or [j,i,'BEGINDASH'] in bonds_:
                    mol.RemoveBond(i, j)
                    mol.AddBond(i, j, Chem.BondType.SINGLE)
                    mol.GetBondBetweenAtoms(i, j).SetBondDir(bond_dirs['BEGINDASH'])
                elif [i,j,'ENDDOWNRIGHT'] in bonds_ or [j,i,'ENDDOWNRIGHT'] in bonds_:
                    mol.RemoveBond(i, j)
                    mol.AddBond(i, j, Chem.BondType.SINGLE)
                    mol.GetBondBetweenAtoms(i, j).SetBondDir(bond_dirs['ENDDOWNRIGHT'])
            Chem.AssignChiralTypesFromBondDirs(mol)
            Chem.AssignStereochemistry(mol, force=True,cleanIt=True)
        for atom in mol.GetAtoms():
            if atom.GetSymbol() != "C":
                atom.SetChiralTag(Chem.rdchem.ChiralType.CHI_UNSPECIFIED)
        mol = mol.GetMol()
    except Exception as e:
        print(e)
    return mol
def _parse_tokens(tokens: list):
    elements = []
    i = 0
    j = 0
    while i < len(tokens):
        if tokens[i] == '(':
            while j < len(tokens) and tokens[j] != ')':
                j += 1
            elt = _parse_tokens(tokens[i + 1:j])
        else:
            elt = tokens[i]
        j += 1
        if j < len(tokens) and tokens[j].isnumeric():
            num = int(tokens[j])
            j += 1
        else:
            num = 1
        elements.append((elt, num))
        i = j
    return elements
def _parse_formula(formula: str):
    tokens = FORMULA_REGEX.findall(formula)
    return _parse_tokens(tokens)
def _expand_carbon(elements: list):
    expanded = []
    i = 0
    while i < len(elements):
        elt, num = elements[i]
        if elt == 'C' and num > 1 and i + 1 < len(elements):
            next_elt, next_num = elements[i + 1]
            quotient, remainder = next_num // num, next_num % num
            for _ in range(num):
                expanded.append('C')
                for _ in range(quotient):
                    expanded.append(next_elt)
            for _ in range(remainder):
                expanded.append(next_elt)
            i += 2
        elif isinstance(elt, list):
            new_elt = _expand_carbon(elt)
            for _ in range(num):
                expanded.append(new_elt)
            i += 1
        else:
            for _ in range(num):
                expanded.append(elt)
            i += 1
    return expanded
def _expand_abbreviation(abbrev):
    if abbrev in ABBREVIATIONS:
        return ABBREVIATIONS[abbrev].smiles
    if abbrev in RGROUP_SYMBOLS or (abbrev[0] == 'R' and abbrev[1:].isdigit()):
        if abbrev[1:].isdigit():
            return f'[{abbrev[1:]}*]'
        return '*'
    return f'[{abbrev}]'
def _get_bond_symb(bond_num):
    if bond_num == 0:
        return '.'
    if bond_num == 1:
        return ''
    if bond_num == 2:
        return '='
    if bond_num == 3:
        return '#'
    return ''
def _condensed_formula_list_to_smiles(formula_list, start_bond, end_bond=None, direction=None):
    if direction is None:
        num_trials = 1
        for dir_choice in [1, -1]:
            smiles, bonds_left, trials, success = _condensed_formula_list_to_smiles(formula_list, start_bond, end_bond, dir_choice)
            num_trials += trials
            if success:
                return smiles, bonds_left, num_trials, success
        return None, None, num_trials, False
    assert direction == 1 or direction == -1
    def dfs(smiles, bonds_left, cur_idx, add_idx):
        num_trials = 1
        if (direction == 1 and add_idx == len(formula_list)) or (direction == -1 and add_idx == -1):
            if end_bond is not None and end_bond != bonds_left:
                return smiles, bonds_left, num_trials, False
            return smiles, bonds_left, num_trials, True
        if bonds_left <= 0:
            return smiles, bonds_left, num_trials, False
        to_add = formula_list[add_idx]  
        if isinstance(to_add, list):  
            if bonds_left > 1:
                add_str, val, trials, success = _condensed_formula_list_to_smiles(to_add, 1, None, direction)
                if val > 0:
                    add_str = _get_bond_symb(val + 1) + add_str
                num_trials += trials
                if not success:
                    return smiles, bonds_left, num_trials, False
                result = dfs(smiles + f'({add_str})', bonds_left - 1, cur_idx, add_idx + direction)
            else:
                add_str, bonds_left, trials, success = _condensed_formula_list_to_smiles(to_add, 1, None, direction)
                num_trials += trials
                if not success:
                    return smiles, bonds_left, num_trials, False
                result = dfs(smiles + add_str, bonds_left, add_idx, add_idx + direction)
            smiles, bonds_left, trials, success = result
            num_trials += trials
            return smiles, bonds_left, num_trials, success
        for val in VALENCES.get(to_add, [1]):  
            add_str = _expand_abbreviation(to_add)  
            if bonds_left > val:  
                if cur_idx >= 0:
                    add_str = _get_bond_symb(val) + add_str
                result = dfs(smiles + f'({add_str})', bonds_left - val, cur_idx, add_idx + direction)
            else:  
                if cur_idx >= 0:
                    add_str = _get_bond_symb(bonds_left) + add_str
                result = dfs(smiles + add_str, val - bonds_left, add_idx, add_idx + direction)
            trials, success = result[2:]
            num_trials += trials
            if success:
                return result[0], result[1], num_trials, success
            if num_trials > 10000:
                break
        return smiles, bonds_left, num_trials, False
    cur_idx = -1 if direction == 1 else len(formula_list)
    add_idx = 0 if direction == 1 else len(formula_list) - 1
    return dfs('', start_bond, cur_idx, add_idx)
def get_smiles_from_symbol(symbol, mol, atom, bonds):
    if symbol in ABBREVIATIONS:
        return ABBREVIATIONS[symbol].smiles
    if len(symbol) > 20:
        return None
    total_bonds = int(sum([bond.GetBondTypeAsDouble() for bond in bonds]))
    formula_list = _expand_carbon(_parse_formula(symbol))
    smiles, bonds_left, num_trails, success = _condensed_formula_list_to_smiles(formula_list, total_bonds, None)
    if success:
        return smiles
    return None
def _replace_functional_group(smiles):
    smiles = smiles.replace('<unk>', 'C')
    for i, r in enumerate(RGROUP_SYMBOLS):
        symbol = f'[{r}]'
        if symbol in smiles:
            if r[0] == 'R' and r[1:].isdigit():
                smiles = smiles.replace(symbol, f'[{int(r[1:])}*]')
            else:
                smiles = smiles.replace(symbol, '*')
    tokens = atomwise_tokenizer(smiles)
    new_tokens = []
    mappings = {}  
    isotope = 50
    for token in tokens:
        if token[0] == '[':
            if token[1:-1] in ABBREVIATIONS or Chem.AtomFromSmiles(token) is None:
                while f'[{isotope}*]' in smiles or f'[{isotope}*]' in new_tokens:
                    isotope += 1
                placeholder = f'[{isotope}*]'
                mappings[isotope] = token[1:-1]
                new_tokens.append(placeholder)
                continue
        new_tokens.append(token)
    smiles = ''.join(new_tokens)
    return smiles, mappings
def convert_smiles_to_mol(smiles):
    if smiles is None or smiles == '':
        return None
    try:
        mol = Chem.MolFromSmiles(smiles)
    except:
        return None
    return mol
BOND_TYPES = {1: Chem.rdchem.BondType.SINGLE, 2: Chem.rdchem.BondType.DOUBLE, 3: Chem.rdchem.BondType.TRIPLE}
def _expand_functional_group(mol, mappings, debug=False):
    def _need_expand(mol, mappings):
        return any([len(Chem.GetAtomAlias(atom)) > 0 for atom in mol.GetAtoms()]) or len(mappings) > 0
    if _need_expand(mol, mappings):
        mol_w = Chem.RWMol(mol)
        num_atoms = mol_w.GetNumAtoms()
        for i, atom in enumerate(mol_w.GetAtoms()):  
            atom.SetNumRadicalElectrons(0)
        atoms_to_remove = []
        for i in range(num_atoms):
            atom = mol_w.GetAtomWithIdx(i)
            if atom.GetSymbol() == '*':
                symbol = Chem.GetAtomAlias(atom)
                isotope = atom.GetIsotope()
                if isotope > 0 and isotope in mappings:
                    symbol = mappings[isotope]
                if not (isinstance(symbol, str) and len(symbol) > 0):
                    continue
                if symbol in RGROUP_SYMBOLS:
                    continue
                bonds = atom.GetBonds()
                sub_smiles = get_smiles_from_symbol(symbol, mol_w, atom, bonds)
                mol_r = convert_smiles_to_mol(sub_smiles)
                if mol_r is None:
                    atom.SetIsotope(0)
                    continue
                adjacent_indices = [bond.GetOtherAtomIdx(i) for bond in bonds]
                for adjacent_idx in adjacent_indices:
                    mol_w.RemoveBond(i, adjacent_idx)
                adjacent_atoms = [mol_w.GetAtomWithIdx(adjacent_idx) for adjacent_idx in adjacent_indices]
                for adjacent_atom, bond in zip(adjacent_atoms, bonds):
                    adjacent_atom.SetNumRadicalElectrons(int(bond.GetBondTypeAsDouble()))
                bonding_atoms_w = adjacent_indices
                bonding_atoms_r = [mol_w.GetNumAtoms()]
                for atm in mol_r.GetAtoms():
                    if atm.GetNumRadicalElectrons() and atm.GetIdx() > 0:
                        bonding_atoms_r.append(mol_w.GetNumAtoms() + atm.GetIdx())
                combo = Chem.CombineMols(mol_w, mol_r)
                mol_w = Chem.RWMol(combo)
                for atm in bonding_atoms_w:
                    bond_order = mol_w.GetAtomWithIdx(atm).GetNumRadicalElectrons()
                    mol_w.AddBond(atm, bonding_atoms_r[0], order=BOND_TYPES[bond_order])
                for atm in bonding_atoms_w:
                    mol_w.GetAtomWithIdx(atm).SetNumRadicalElectrons(0)
                for atm in bonding_atoms_r:
                    mol_w.GetAtomWithIdx(atm).SetNumRadicalElectrons(0)
                atoms_to_remove.append(i)
        atoms_to_remove.sort(reverse=True)
        for i in atoms_to_remove:
            mol_w.RemoveAtom(i)
        smiles = Chem.MolToSmiles(mol_w)
        mol = mol_w.GetMol()
    else:
        smiles = Chem.MolToSmiles(mol)
    return smiles, mol
def _convert_graph_to_smiles(coords, symbols, edges, image=None, debug=False):
    mol = Chem.RWMol()
    n = len(symbols)
    ids = []
    for i in range(n):
        symbol = symbols[i]
        if symbol[0] == '[':
            symbol = symbol[1:-1]
        if symbol in RGROUP_SYMBOLS:
            atom = Chem.Atom("*")
            if symbol[0] == 'R' and symbol[1:].isdigit():
                atom.SetIsotope(int(symbol[1:]))
            Chem.SetAtomAlias(atom, symbol)
        elif symbol in ABBREVIATIONS:
            atom = Chem.Atom("*")
            Chem.SetAtomAlias(atom, symbol)
        else:
            try:  
                atom = Chem.AtomFromSmiles(symbols[i])
                atom.SetChiralTag(Chem.rdchem.ChiralType.CHI_UNSPECIFIED)
            except:  
                atom = Chem.Atom("*")
                Chem.SetAtomAlias(atom, symbol)
        if atom.GetSymbol() == '*':
            atom.SetProp('molFileAlias', symbol)
        idx = mol.AddAtom(atom)
        assert idx == i
        ids.append(idx)
    for i in range(n):
        for j in range(i + 1, n):
            if edges[i][j] == 1:
                mol.AddBond(ids[i], ids[j], Chem.BondType.SINGLE)
            elif edges[i][j] == 2:
                mol.AddBond(ids[i], ids[j], Chem.BondType.DOUBLE)
            elif edges[i][j] == 3:
                mol.AddBond(ids[i], ids[j], Chem.BondType.TRIPLE)
            elif edges[i][j] == 4:
                mol.AddBond(ids[i], ids[j], Chem.BondType.AROMATIC)
            elif edges[i][j] == 5:
                mol.AddBond(ids[i], ids[j], Chem.BondType.SINGLE)
                mol.GetBondBetweenAtoms(ids[i], ids[j]).SetBondDir(Chem.BondDir.BEGINWEDGE)
            elif edges[i][j] == 6:
                mol.AddBond(ids[i], ids[j], Chem.BondType.SINGLE)
                mol.GetBondBetweenAtoms(ids[i], ids[j]).SetBondDir(Chem.BondDir.BEGINDASH)
    pred_smiles = '<invalid>'
    try:
        if image is not None:
            height, width, _ = image.shape
            ratio = width / height
            coords = [[x * ratio * 10, y * 10] for x, y in coords]
        mol = _verify_chirality(mol, coords, symbols, edges, debug)
        pred_molblock = Chem.MolToMolBlock(mol)
        pred_smiles, mol = _expand_functional_group(mol, {}, debug)
        success = True
    except Exception as e:
        if debug:
            print(traceback.format_exc())
        pred_molblock = ''
        success = False
    if debug:
        return pred_smiles, pred_molblock, mol, success
    return pred_smiles, pred_molblock, success
def convert_graph_to_smiles(coords, symbols, edges, images=None, num_workers=16):
    if images is None:
        args_zip = zip(coords, symbols, edges)
    else:
        args_zip = zip(coords, symbols, edges, images)
    if num_workers <= 1:
        results = itertools.starmap(_convert_graph_to_smiles, args_zip)
        results = list(results)
    else:
        with multiprocessing.Pool(num_workers) as p:
            results = p.starmap(_convert_graph_to_smiles, args_zip, chunksize=128)
    smiles_list, molblock_list, success = zip(*results)
    r_success = np.mean(success)
    return smiles_list, molblock_list, r_success
def _postprocess_smiles(smiles, coords=None, symbols=None, edges=None, molblock=False, debug=False):
    if type(smiles) is not str or smiles == '':
        return '', False
    mol = None
    pred_molblock = ''
    try:
        pred_smiles = smiles
        pred_smiles, mappings = _replace_functional_group(pred_smiles)
        if coords is not None and symbols is not None and edges is not None:
            pred_smiles = pred_smiles.replace('@', '').replace('/', '').replace('\\', '')
            mol = Chem.RWMol(Chem.MolFromSmiles(pred_smiles, sanitize=False))
            mol = _verify_chirality(mol, coords, symbols, edges, debug)
        else:
            mol = Chem.MolFromSmiles(pred_smiles, sanitize=False)
        if molblock:
            pred_molblock = Chem.MolToMolBlock(mol)
        pred_smiles, mol = _expand_functional_group(mol, mappings)
        success = True
    except Exception as e:
        if debug:
            print(traceback.format_exc())
        pred_smiles = smiles
        pred_molblock = ''
        success = False
    if debug:
        return pred_smiles, pred_molblock, mol, success
    return pred_smiles, pred_molblock, success
def postprocess_smiles(smiles, coords=None, symbols=None, edges=None, molblock=False, num_workers=16):
    with multiprocessing.Pool(num_workers) as p:
        if coords is not None and symbols is not None and edges is not None:
            results = p.starmap(_postprocess_smiles, zip(smiles, coords, symbols, edges), chunksize=128)
        else:
            results = p.map(_postprocess_smiles, smiles, chunksize=128)
    smiles_list, molblock_list, success = zip(*results)
    r_success = np.mean(success)
    return smiles_list, molblock_list, r_success
def _keep_main_molecule(smiles, debug=False):
    try:
        mol = Chem.MolFromSmiles(smiles)
        frags = Chem.GetMolFrags(mol, asMols=True)
        if len(frags) > 1:
            num_atoms = [m.GetNumAtoms() for m in frags]
            main_mol = frags[np.argmax(num_atoms)]
            smiles = Chem.MolToSmiles(main_mol)
    except Exception as e:
        if debug:
            print(traceback.format_exc())
    return smiles
def keep_main_molecule(smiles, num_workers=16):
    with multiprocessing.Pool(num_workers) as p:
        results = p.map(_keep_main_molecule, smiles, chunksize=128)
    return results