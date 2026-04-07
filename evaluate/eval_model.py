import os,sys
import argparse
model_usedpath='/recovery/bo/pys/I2M'
sys.path.append(model_usedpath)
home='/recovery/bo/pys/I2M'
bmd=f'/recovery/bo/pys/I2M/weights/I2M_R4.onnx'
parser = argparse.ArgumentParser()
parser.add_argument('--config', '-c', type=str, default=f'{home}/configs/moldetr/moldetr_r50vd_6x_coco.yml')
parser.add_argument('--resume', '-r', type=str, default=f'{bmd}')
parser.add_argument('--tuning', '-t', type=str,)
parser.add_argument('--test-only',default=True,)
parser.add_argument('--amp', default=False,)
parser.add_argument('--dataname', '-da', type=str, default=None)
parser.add_argument('--gpuid', '-gi', type=str, default=None)
parser.add_argument('--number', '-n', type=str, default=None)
args, unknown = parser.parse_known_args()
print(args)
if args.gpuid:
    os.environ['CUDA_VISIBLE_DEVICES'] = f'{args.gpuid}'
else:
    os.environ['CUDA_VISIBLE_DEVICES'] = '4'
parralel_n=2
os.environ["OMP_NUM_THREADS"] = f"{parralel_n}"       
os.environ["OPENBLAS_NUM_THREADS"] = f"{parralel_n}"  
os.environ["MKL_NUM_THREADS"] = f"{parralel_n}"       
os.environ["VECLIB_MAXIMUM_THREADS"] = f"{parralel_n}"  
os.environ["NUMEXPR_NUM_THREADS"] = f"{parralel_n}"   
import shutil
import pandas as pd
print(__file__)
cur_dir = os.path.dirname(os.path.abspath(__file__))
print(cur_dir)
python_path=cur_dir
sys.path.append(python_path)
from src.solver.det_engine import *
import cv2
import sys,copy
import torchvision
import torch
import tqdm
import onnxruntime as ort
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
from det_engine import N_C_H_expand, C_H_expand,C_H_expand2, C_F_expand, formula_regex, RTDETRPostProcessor
from det_engine import SmilesEvaluator, molfpsim
from utils import calculate_iou,adjust_bbox1
from det_engine import normalize_ocr_text, check_and_fix_valence, rdkit_canonicalize_smiles
from det_engine import is_valid_chem_text,select_chem_expression
from det_engine import ABBREVIATIONS,remove_SP, select_longest_smiles
from det_engine import molExpanding,remove_bond_directions_if_no_chiral
from det_engine import (comparing_smiles,comparing_smiles2, remove_SP, expandABB,               ELEMENTS,        ABBREVIATIONS)
import rdkit 
from rdkit import Chem
from rdkit.Chem import Draw, AllChem
from rdkit import DataStructs
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
print("CUDA available:", torch.cuda.is_available())
print("Number of GPUs:", torch.cuda.device_count())
print("Device name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU")
def bbox_area(bbox):
    x1, y1, x2, y2 = bbox
    return (x2 - x1) * (y2 - y1)
def mol_idx( mol ):
    atoms = mol.GetNumAtoms()
    for idx in range( atoms ):
        mol.GetAtomWithIdx( idx ).SetProp( 'molAtomMapNumber', str( mol.GetAtomWithIdx( idx ).GetIdx() ) )
    return mol
def mol_idx_del(mol):
    atoms = mol.GetNumAtoms()
    for idx in range(atoms):
        atom = mol.GetAtomWithIdx(idx)
        if atom.HasProp('molAtomMapNumber'):  
            atom.ClearProp('molAtomMapNumber')  
    return mol
def is_contained_in(bbox_small, bbox_large):
    x_min_s, y_min_s, x_max_s, y_max_s = bbox_small
    x_min_l, y_min_l, x_max_l, y_max_l = bbox_large
    return (x_min_s >= x_min_l and x_max_s <= x_max_l and 
            y_min_s >= y_min_l and y_max_s <= y_max_l)
def NoRadical_Smi(smi):
    aa=Chem.MolFromSmiles(smi)
    for atom in aa.GetAtoms():
        if atom.GetNumRadicalElectrons() > 0:  
            atom.SetNumRadicalElectrons(0)  
            atom.SetNumExplicitHs(atom.GetTotalValence() - atom.GetExplicitValence())
    san_before=Chem.MolToSmiles(aa)
    return san_before
def parse_charge(charge_str):
    if charge_str.endswith('+'):
        return int(charge_str[:-1]) if charge_str[:-1] else 1  
    elif charge_str.endswith('-'):
        return -int(charge_str[:-1]) if charge_str[:-1] else -1  
    else :
        return int(charge_str)
def set_bondDriection(rwmol_,bondWithdirct):
    chiral_center_ids = Chem.FindMolChiralCenters(rwmol_, includeUnassigned=True)
    chirai_ai2sterolab=dict()
    if len(chiral_center_ids)>0:
        chirai_ai2sterolab={ai:stero_lab for ai, stero_lab in chiral_center_ids }
    for bi, binfo in bondWithdirct.items():
        atom1_idx, atom2_idx, bond_type, score, w_d = binfo
        bt= rwmol_.GetBondBetweenAtoms(atom1_idx, atom2_idx)
        current_begin = bt.GetBeginAtomIdx()
        current_end = bt.GetEndAtomIdx()
        if w_d=='wdge':
            bond_dir_=rdchem.BondDir.BEGINWEDGE
            reverse_dir = rdchem.BondDir.BEGINDASH 
        elif w_d=='dash':
            bond_dir_=rdchem.BondDir.BEGINDASH
            reverse_dir = rdchem.BondDir.BEGINWEDGE 
        if atom1_idx in chirai_ai2sterolab.keys():
            if current_begin == atom1_idx:
                bt.SetBondDir(bond_dir_)
                print(f'atom1_idx dir')
            else:
                bt.SetBondDir(reverse_dir)
                print(f'atom1_idx reverse_dir')
        elif atom2_idx in chirai_ai2sterolab.keys():
            if current_begin == atom2_idx:
                bt.SetBondDir(bond_dir_)
                print(f'atom2_idx dir {bond_dir_} {reverse_dir}')
            else:
                rwmol_.RemoveBond(current_begin, current_end)
                rwmol_.AddBond(current_end, current_begin, bt.GetBondType())
                bond = rwmol_.GetBondBetweenAtoms(current_end, current_begin)
                bond.SetBondDir(bond_dir_)
                print(f'atom2_idx reverse_dir {bond_dir_} {reverse_dir}')
        else:
            print('bond stro not with chiral atom???, will ignore this stero bond infors')
            print(f"{[bi, bond_dir_, current_begin,current_end]}")
        return rwmol_
atom_labels = [0,1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
bond_labels = [13,14,15,16,17,18]
charge_labels=[19,20,21,22,23]
idx_to_labels={0:'other',1:'C',2:'O',3:'N',4:'Cl',5:'Br',6:'S',7:'F',8:'B',
                    9:'I',10:'P',11:'H',12:'Si',
                    13:'single',14:'wdge',15:'dash',
                    16:'=',17:'#',18:':',
                    19:'-4',20:'-2',
                    21:'-1',
                    22:'+1',
                    23:'+2',
                    }
lab2idx={ v:k for k,v in idx_to_labels.items()}   
bond_labels_symb={idx_to_labels[i] for i in bond_labels}                 
bond_dirs = {'NONE':    Chem.rdchem.BondDir.NONE,
            'ENDUPRIGHT':   Chem.rdchem.BondDir.ENDUPRIGHT,
            'BEGINWEDGE':   Chem.rdchem.BondDir.BEGINWEDGE,
            'BEGINDASH':    Chem.rdchem.BondDir.BEGINDASH,
            'ENDDOWNRIGHT': Chem.rdchem.BondDir.ENDDOWNRIGHT,
            }
import pandas as pd
from typing import Iterable, List
from PIL import Image
import json,re
from scipy.spatial import cKDTree, KDTree
import numpy as np
from rdkit import Chem
from paddleocr import PaddleOCR
from rdkit.Chem import rdchem, RWMol, CombineMols
def bbox2shapes(bboxes, classes, lab2idx):
    shapes = []
    for bbox, label in zip(bboxes, classes):
        x1, y1, x2, y2 = bbox
        if label not in lab2idx : 
            label='other'
        shape = {
            "kie_linking": [],
            "label": label,
            "score": 1.0,
            "points": [
                [x1, y1],  
                [x2, y1],  
                [x2, y2],  
                [x1, y2]   
            ],
            "group_id": None,
            "description": None,
            "difficult": False,
            "shape_type": "rectangle",
            "flags": None,
            "attributes": {}
        }
        shapes.append(shape)
    return shapes
def get_longest_part(smi_string):
    if '.' in smi_string:  
        parts = smi_string.split('.')  
        longest_part = max(parts, key=len)  
        return longest_part
    else:
        return smi_string  
def split_output_by_numeric_classes(output):
    numeric_output = {key: [] for key in output.keys()}
    non_numeric_output = {key: [] for key in output.keys()}
    for i in range(len(output['pred_classes'])):
        class_name = output['pred_classes'][i]
        if re.fullmatch(r'^[+-]?\d+[+-]?$', class_name):
            target_dict = numeric_output
        else:
            target_dict = non_numeric_output
        for key in output.keys():
            target_dict[key].append(output[key][i])
    return numeric_output, non_numeric_output
def convert_shapes_to_output(json_data):
    output = {
        'bbox': [],
        'bbox_centers': [],
        'scores': [],
        'pred_classes': []
    }
    for shape in json_data['shapes']:
        points = shape['points']
        x_coords = [p[0] for p in points]
        y_coords = [p[1] for p in points]
        bbox = [min(x_coords), min(y_coords), max(x_coords), max(y_coords)]
        center_x = (bbox[0] + bbox[2]) / 2
        center_y = (bbox[1] + bbox[3]) / 2
        score = shape.get('score', 1.0)
        pred_class = shape['label']
        output['bbox'].append(bbox)
        output['bbox_centers'].append([center_x, center_y])
        output['scores'].append(score)
        output['pred_classes'].append(pred_class)
    return output
def getJsonData(src_json):
    with open(src_json, 'r') as f:
            coco_data = json.load(f)
    return coco_data
def replace_cg_notation(astr):
    def replacer(match):
        h_count = int(match.group(1))
        c_count = (h_count - 1) // 2
        return f'C{c_count}H{h_count}'
    return re.sub(r'CgH(\d+)', replacer, astr)
def viewcheck(image_path,bbox,color='red'):
    image = Image.open(image_path)
    image_array = np.array(image)
    plt.figure(figsize=(5, 4))  
    plt.imshow(image_array)  
    bbox = np.array(bbox)
    x_coords = (bbox[:, 0]+bbox[:, 2])*0.5
    y_coords =( bbox[:, 1]+bbox[:, 3])*0.5
    plt.scatter(x_coords, y_coords, c=color, s=50, label='Atom Centers', edgecolors='black')
    for i, (x, y) in enumerate(zip(x_coords, y_coords)):
        plt.text(x, y, f'a {i}', fontsize=12, color=color, ha='center', va='bottom')
bclass_simple={"single":'-', "wdge":'w', "dash":'--', 
                "=":'=', "#":"#", ":":"aro"}
def viewcheck_b(image_path,bbox,bclass,color='red',figsize=(5,4)):
    image = Image.open(image_path)
    image_array = np.array(image)
    plt.figure(figsize=figsize)  
    plt.imshow(image_array)  
    bbox = np.array(bbox)
    x_coords = (bbox[:, 0]+bbox[:, 2])*0.5
    y_coords =( bbox[:, 1]+bbox[:, 3])*0.5
    plt.scatter(x_coords, y_coords, c=color, s=50, label='bond Centers', edgecolors='black')
    for i, (x, y) in enumerate(zip(x_coords, y_coords)):
        simpl_b=bclass_simple[bclass[i]]
        plt.text(x, y, f'b{i}{simpl_b}', fontsize=12, color=color, ha='center', va='bottom')    
def anchor_draw(image_path, bond_bbox):
    image = Image.open(image_path)
    image_array = np.array(image)
    _margin = 3
    all_anchor_positions = []
    all_oposite_anchor_positions = []
    for bi, bbox in enumerate(bond_bbox):
        anchor_positions = (np.array(bbox) + [_margin, _margin, -_margin, -_margin]).reshape([2, -1])
        oposite_anchor_positions = anchor_positions.copy()
        oposite_anchor_positions[:, 1] = oposite_anchor_positions[:, 1][::-1]
        anchor_positions = np.concatenate([anchor_positions, oposite_anchor_positions])
        all_anchor_positions.append(anchor_positions[:2])  
        all_oposite_anchor_positions.append(anchor_positions[2:])  
    all_anchor_positions = np.array(all_anchor_positions).reshape(-1, 2)
    all_oposite_anchor_positions = np.array(all_oposite_anchor_positions).reshape(-1, 2)
    plt.figure(figsize=(10, 8))
    plt.imshow(image_array)
    plt.scatter(all_anchor_positions[:, 0], all_anchor_positions[:, 1], c='red', s=50, label='Anchor Positions', edgecolors='black')
    for i, (x, y) in enumerate(all_anchor_positions):
        plt.text(x, y, f'B{int(i/2)}:{i%2}', fontsize=10, color='white', ha='center', va='bottom')
    plt.title('Anchor Positions (Upper Left, Lower Right)')
    plt.legend()
    plt.axis('off')
    plt.savefig('anchor_positions.png')
    plt.figure(figsize=(10, 8))
    plt.imshow(image_array)
    plt.scatter(all_oposite_anchor_positions[:, 0], all_oposite_anchor_positions[:, 1], c='blue', s=50, label='Opposite Anchor Positions', edgecolors='black')
    for i, (x, y) in enumerate(all_oposite_anchor_positions):
        plt.text(x, y, f'B{int(i/2)}:{i%2}', fontsize=10, color='white', ha='center', va='bottom')
    plt.title('Opposite Anchor Positions (Lower Left, Upper Right)')
    plt.legend()
    plt.axis('off')
    plt.savefig('Opposite_anchor_positions.png')
def get_corners(bbox):
    x_min, y_min, x_max, y_max = bbox
    return np.array([
        [x_min, y_min], [x_max, y_min],  
        [x_min, y_max], [x_max, y_max]   
    ])
def find_nearest_atom(bond_corners, atom_bboxes, exclude_idx=None):
    min_dist = float('inf')
    nearest_idx = None
    for i, atom_bbox in enumerate(atom_bboxes):
        if exclude_idx is not None and i in exclude_idx:
            continue
        atom_corners = get_corners(atom_bbox)
        for bc in bond_corners:
            for ac in atom_corners:
                dist = np.sqrt((bc[0] - ac[0])**2 + (bc[1] - ac[1])**2)
                if dist < min_dist:
                    min_dist = dist
                    nearest_idx = i
    return nearest_idx, min_dist
def get_min_distance_to_atom_box(vertices, atom_bboxes, exclude_idx=None):
    min_dist = float('inf')
    closest_atom_idx = -1
    for i, ab in enumerate(atom_bboxes):
        if exclude_idx is not None and i in  exclude_idx:
            continue
        ab_vertices = np.array([[ab[0], ab[1]], [ab[2], ab[3]], [ab[0], ab[3]], [ab[2], ab[1]]])
        for v in vertices:
            for av in ab_vertices:
                dist = np.linalg.norm(v - av)
                if dist < min_dist:
                    min_dist = dist
                    closest_atom_idx = i
    return min_dist, closest_atom_idx
def boxes_overlap(box1, box2):
    x1, y1, x2, y2 = box1
    x3, y3, x4, y4 = box2
    return not (x2 < x3 or x4 < x1 or y2 < y3 or y4 < y1)
def min_corner_distance(box1, box2):
    corners1 = [[box1[0], box1[1]], [box1[2], box1[3]], [box1[0], box1[3]], [box1[2], box1[1]]]
    corners2 = [[box2[0], box2[1]], [box2[2], box2[3]], [box2[0], box2[3]], [box2[2], box2[1]]]
    min_dist = float('inf')
    for c1 in corners1:
        for c2 in corners2:
            dist = np.sqrt((c1[0] - c2[0])**2 + (c1[1] - c2[1])**2)
            min_dist = min(min_dist, dist)
    return min_dist
def clear_directory(path):
    if os.path.exists(path):
        print(f"Clearing contents of: {path}")
        for filename in os.listdir(path):
            file_path = os.path.join(path, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)  
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)  
            except Exception as e:
                print(f'Failed to delete {file_path}. Reason: {e}')
    else:
        print(f"Directory does not exist: {path}")
def NHR_string(text):
    pattern1 = r'NHR\d'
    pattern2 = r'RHN[0-9a-z]+'
    pattern3 = r'R[0-9a-z]+NH'
    if re.search(pattern1, text):
        text='NH*'
    elif re.search(pattern2, text):
        text='NH*'
    elif re.search(pattern3, text):
        text='NH*'
    return text
def preprocess_atom_boxes(atom_centers, atom_bbox, size_threshold_factor=2.5, min_subboxes=2):
    areas = [(bbox[2] - bbox[0]) * (bbox[3] - bbox[1]) for bbox in atom_bbox]
    if len(areas) > 2:
        sorted_areas = sorted(areas)
        avg_area = np.mean(sorted_areas[1:-1])  
    else:
        avg_area = np.mean(areas) if areas else 1.0
    new_atom_centers = []
    new_atom_bbox = []
    original_to_subbox = {}  
    subbox_to_original = {}  
    new_idx = 0
    for i, (bbox, center) in enumerate(zip(atom_bbox, atom_centers)):
        area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        if area > avg_area * size_threshold_factor:
            num_subboxes = max(min_subboxes, int(round(area / avg_area)))
            width = bbox[2] - bbox[0]
            height = bbox[3] - bbox[1]
            if width >= height:
                sub_width = width / num_subboxes
                subboxes = [
                    [bbox[0] + j * sub_width, bbox[1], bbox[0] + (j + 1) * sub_width, bbox[3]]
                    for j in range(num_subboxes)
                ]
            else:
                sub_height = height / num_subboxes
                subboxes = [
                    [bbox[0], bbox[1] + j * sub_height, bbox[2], bbox[1] + (j + 1) * sub_height]
                    for j in range(num_subboxes)
                ]
            sub_centers = [
                [(subbox[0] + subbox[2]) / 2, (subbox[1] + subbox[3]) / 2]
                for subbox in subboxes
            ]
            new_atom_bbox.extend(subboxes)
            new_atom_centers.extend(sub_centers)
            original_to_subbox[i] = list(range(new_idx, new_idx + num_subboxes))
            for j in range(num_subboxes):
                subbox_to_original[new_idx + j] = i
            new_idx += num_subboxes
        else:
            new_atom_bbox.append(bbox)
            new_atom_centers.append(center)
            original_to_subbox[i] = [new_idx]
            subbox_to_original[new_idx] = i
            new_idx += 1
    return np.array(new_atom_centers), new_atom_bbox, original_to_subbox, subbox_to_original
other2ppsocr=True
if other2ppsocr:
    ocr = PaddleOCR(
    use_angle_cls=True,
    lang='latin',use_space_char=True,use_debug=False,
    use_gpu=True if cv2.cuda.getCudaEnabledDeviceCount() > 0 else False)
    ocr2 = ocr2 = PaddleOCR(use_angle_cls=True,use_gpu =False,use_debug=False,
                rec_algorithm='SVTR_LCNet', rec_model_dir='//home/bo/.paddleocr/whl/rec/en/en_PP-OCRv4_rec_infer',
                lang="en") 
box_thresh=0.5
useocr=True
box_matter=0
getacc=True
getfpsim=True
visual_check=False
da='acs'
da='CLEF'
da='UOB'
da='USPTO'
da='chemvlocr'
if args.dataname:
    da=args.dataname
ac_b=False    
ac_b_smilesnotsame_writJson=True
if ac_b:
    view_check_dir=f"D:\RPA\codes_share\wsl_\X-AnyLabeling\\need2check\\view_check_{da}\\failed"
    view_dirac=f"{view_check_dir}/{da}_ac"
    view_dirb=f"{view_check_dir}/{da}_b"
    dst_dirac =view_dirac
    dst_dirb =view_dirb
src_dir='/recovery/bo/pys/i2m_data/real'
src_file = os.path.join(src_dir, f"{da}.csv")
df = pd.read_csv(src_file)
print(f"src_file:\n{src_file}")
view_check_base = os.path.join(src_dir, f"view_check_{da}", "v3")
target_da_ac_dir = os.path.join(view_check_base, f"{da}_ac")
if not os.path.exists(target_da_ac_dir):
    os.makedirs(target_da_ac_dir)
    print(f"Created directory: {target_da_ac_dir}")
source_da_dir = os.path.join(src_dir, da)
if os.path.exists(source_da_dir):
    copied_count = 0
    for img_file in os.listdir(source_da_dir):
        if img_file.endswith('.png'):
            src_path = os.path.join(source_da_dir, img_file)
            dst_path = os.path.join(target_da_ac_dir, img_file)
            if not os.path.exists(dst_path):
                shutil.copy2(src_path, dst_path)
                copied_count += 1
    print(f"Copied {copied_count} images from {source_da_dir} to {target_da_ac_dir}")
else:
    print(f"Warning: Source directory {source_da_dir} does not exist")
view_check_dir2 = os.path.join(src_dir, f"view_check_{da}", "v3")
N=1
if args.number:
    N=int(args.number)
view_dirac2 = os.path.join(view_check_dir2, f"{da}_ac")
view_dirb2 = os.path.join(view_check_dir2, f"{da}_b")
view_dirac_tmp = os.path.join(view_check_dir2, f"{da}_actmp")
view_dirac_tmp_debug=True
if ac_b:
    need2mkdir=[view_check_dir,view_dirac, view_dirb, view_check_dir2,view_dirac2, view_dirb2]
else:
    need2mkdir=[ view_check_dir2,view_dirac2, view_dirb2,view_dirac_tmp]
for dir_v in need2mkdir :
    if not os.path.exists(dir_v):
        os.makedirs(dir_v)
df['file_name'] = df['file_path'].str.split('/').str[-1]
df['file_base'] = df['file_name'].str.replace('.png', '', regex=False)
prefilter_none_mol_csv = os.path.join(src_dir, f"{da}_none_mol_images.csv")
prefilter_none_mol_records = []
prefilter_skip_file_names = set()
if os.path.exists(prefilter_none_mol_csv):
    prefilter_none_mol_df = pd.read_csv(prefilter_none_mol_csv)
    if 'file_name' in prefilter_none_mol_df.columns:
        prefilter_none_mol_df['file_name'] = prefilter_none_mol_df['file_name'].astype(str)
        prefilter_skip_file_names = set(prefilter_none_mol_df['file_name'].tolist())
    elif 'file_path' in prefilter_none_mol_df.columns:
        prefilter_none_mol_df['file_name'] = prefilter_none_mol_df['file_path'].astype(str).map(os.path.basename)
        prefilter_skip_file_names = set(prefilter_none_mol_df['file_name'].tolist())
    removed_count = int(df['file_name'].isin(prefilter_skip_file_names).sum())
    if removed_count > 0:
        df = df[~df['file_name'].isin(prefilter_skip_file_names)].copy()
    for record in prefilter_none_mol_df.to_dict('records'):
        record.setdefault('file_name', os.path.basename(str(record.get('file_path', ''))))
        record.setdefault('file_base', str(record['file_name']).replace('.png', ''))
        record.setdefault('reason', 'prefiltered_none_mol')
        record.setdefault('atom1_idx', None)
        record.setdefault('atom2_idx', None)
        record.setdefault('atom2_1_idx', None)
        prefilter_none_mol_records.append(record)
    print(
        f"Prefiltered {removed_count} known none-molecule images from "
        f"{prefilter_none_mol_csv}"
    )
else:
    print(f"No prefilter none-molecule CSV found at: {prefilter_none_mol_csv}")
real_data_dir = os.path.join('/recovery/bo/pys/i2m_data/real', da)
source_ac_dir = os.path.join(src_dir, f"{da}_ac")
target_ac_dir = view_dirac2
copied_from_real = 0
copied_from_local = 0
missing_source_images = []
for i, row in df.iterrows():
    if i<=40616: continue
    png_filename = df.loc[i,'file_name']
    img_name=df.loc[i,'file_name']
    target_path = os.path.join(target_ac_dir, img_name)
    if os.path.exists(target_path):
        continue
    real_src_path = os.path.join(real_data_dir, img_name)
    local_src_path = os.path.join(source_ac_dir, img_name)
    if os.path.exists(real_src_path):
        shutil.copy2(real_src_path, target_path)
        copied_from_real += 1
    elif os.path.exists(local_src_path):
        shutil.copy2(local_src_path, target_path)
        copied_from_local += 1
    else:
        missing_source_images.append(img_name)
print(
    f"Prepared {target_ac_dir}: copied {copied_from_real} from {real_data_dir}, "
    f"{copied_from_local} from {source_ac_dir}"
)
if missing_source_images:
    print(
        f"Warning: {len(missing_source_images)} images were not found in either "
        f"{real_data_dir} or {source_ac_dir}"
    )
    print(f"First missing examples: {missing_source_images[:10]}")
ac_b=False
debug=True
if debug:print(f'get dataframe {len(df)} !!!!')
outcsv_filename=os.path.join(src_dir, f"{da}_OUTPUTwithOCR.csv")
if getacc:
    acc_summary=f"{outcsv_filename}.I2Msummary.txt"
    flogout = open(f'{acc_summary}' , 'w')
    flogout2 = open(f'{outcsv_filename}_acBoxWrong' , 'a')
    failed=[]
    failed_fb=[]
    mydiff=[]
    simRD=0
    simRDlist=[]
    sim=0
    mysum=0
smiles_data = pd.DataFrame({'file_name': [],
                                'SMILESori':[],
                                'SMILESpre':[],
                                'SMILESexp':[],
                                })
none_mol_images = list(prefilter_none_mol_records)
rows_check = df
if args.number:
    rows_check = df.head(N)
miss_file=[]
miss_filejs=[]
debug=True
rt_out=False
if not ac_b:
    view_dirac=view_dirac2
    view_dirb=view_dirb2
    dst_dirac =view_dirac
    dst_dirb =view_dirb
    pngs=[f for f in os.listdir(view_dirac2) if '.png' in f]
    rt_out=True
    view_check_dir3= os.path.join(src_dir, f"view_check_{da}", "v4")
    view_dirac3=f"{view_check_dir3}/{da}_ac"
    view_dirb3=f"{view_check_dir3}/{da}_b"
    for dir_v in [view_check_dir3,view_dirac3, view_dirb3]:
        if not os.path.exists(dir_v):
            os.makedirs(dir_v)
acn=False
bn=False
import torchvision.transforms.v2 as T
def image_to_tensor(image_path,debug=True):
    image = Image.open(image_path)
    w, h = image.size
    if image.mode == "L":
        if debug: print("检测到灰度图像 (1 通道)，转换为 RGB...")
        image = image.convert("RGB")
    elif image.mode != "RGB":
        if debug: print(f"检测到 {image.mode} 模式，转换为 RGB...")
        image = image.convert("RGB")
    transform = T.Compose([
            T.Resize((640, 640)),  
            T.ToTensor(),
            lambda x: x.to(torch.float32),  
        ])
    tensor = transform(image)
    return tensor,w,h
def ouptnp2abc(output,idx_to_labels):
    atom_labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    bond_labels = [13, 14, 15, 16, 17, 18]
    charge_labels = [19, 20, 21, 22, 23]
    atom_mask = np.isin(output['pred_classes'], atom_labels)
    bond_mask = np.isin(output['pred_classes'], bond_labels)
    charge_mask = np.isin(output['pred_classes'], charge_labels)
    output_a = {'bbox': [], 'bbox_centers': [], 'scores': [], 'pred_classes': []}
    output_b = {'bbox': [], 'bbox_centers': [], 'scores': [], 'pred_classes': []}
    output_c = {'bbox': [], 'bbox_centers': [], 'scores': [], 'pred_classes': []}
    if np.any(atom_mask):
        output_a['bbox'] = output['bbox'][atom_mask].tolist()
        output_a['bbox_centers'] = output['bbox_centers'][atom_mask].tolist()
        output_a['scores'] = output['scores'][atom_mask].tolist()
        output_a['pred_classes'] = output['pred_classes'][atom_mask].tolist()
        output_a['pred_classes'] = [idx_to_labels[idx] for idx in output_a['pred_classes']]
    if np.any(bond_mask):
        output_b['bbox'] = output['bbox'][bond_mask].tolist()
        output_b['bbox_centers'] = output['bbox_centers'][bond_mask].tolist()
        output_b['scores'] = output['scores'][bond_mask].tolist()
        output_b['pred_classes'] = output['pred_classes'][bond_mask].tolist()
        output_b['pred_classes'] = [idx_to_labels[idx] for idx in output_b['pred_classes']]
    if np.any(charge_mask):
        output_c['bbox'] = output['bbox'][charge_mask].tolist()
        output_c['bbox_centers'] = output['bbox_centers'][charge_mask].tolist()
        output_c['scores'] = output['scores'][charge_mask].tolist()
        output_c['pred_classes'] = output['pred_classes'][charge_mask].tolist()
        output_c['pred_classes'] = [idx_to_labels[idx] for idx in output_c['pred_classes']]
    return output_a, output_b, output_c
def bbox2center(bbox):
    x_center = (bbox[:, 0] + bbox[:, 2]) / 2
    y_center = (bbox[:, 1] + bbox[:, 3]) / 2
    centers = np.stack((x_center, y_center), axis=1)
    return centers
def extract_ocr_text_score(result):
    if not result:
        return "", 0.0
    try:
        first = result[0][0]
    except Exception:
        return "", 0.0
    if isinstance(first, (list, tuple)):
        text = first[0] if len(first) > 0 else ""
        score = first[1] if len(first) > 1 else 0.0
        try:
            score = float(score)
        except (TypeError, ValueError):
            score = 0.0
        return str(text), score
    if isinstance(first, str):
        return first, 0.0
    return "", 0.0
class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
from src.core import YAMLConfig 
cfg = YAMLConfig(
        args.config,
        resume=args.resume, 
        use_amp=args.amp,
        tuning=args.tuning
    )
print(f"Loading ONNX model from: {args.resume}")
if torch.cuda.is_available():
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    print("Using CUDA for ONNX inference")
else:
    providers = ['CPUExecutionProvider']
    print("Using CPU for ONNX inference")
try:
    onnx_session = ort.InferenceSession(args.resume, providers=providers)
    print("ONNX model loaded successfully")
    print(f"Input names: {onnx_session.get_inputs()}")
    print(f"Output names: {onnx_session.get_outputs()}")
except Exception as e:
    print(f"Error loading ONNX model: {e}")
    raise
postprocessor=RTDETRPostProcessor(classes_dict=idx_to_labels, use_focal_loss=True, num_top_queries=300, remap_mscoco_category=False)
print("when training use this as valdation dataset::",
    cfg.yaml_cfg['val_dataloader']['dataset']['img_folder'],"\n",
cfg.yaml_cfg['val_dataloader']['dataset']['ann_file'])
dfm=0
for id_ in range(len(rows_check)):
    SMILESori=rows_check.iloc[id_].SMILES
    file_base=rows_check.iloc[id_].file_base
    image_path= os.path.join(dst_dirac, f"{file_base}.png")
    print(f"@@@@@@@@@@@@@@@@@@@@@@@ {id_}\n{image_path}\n {SMILESori}")
    img_ori = Image.open(image_path).convert('RGB')
    w_ori, h_ori = img_ori.size  
    scale_x = 1000 / w_ori
    scale_y = 1000 / h_ori
    img_ori_1k = img_ori.resize((1000,1000))
    tensor,w,h = image_to_tensor(image_path)
    tensor=tensor.unsqueeze(0)
    onnx_inputs = onnx_session.get_inputs()
    tensor_np = tensor.cpu().numpy()
    if len(onnx_inputs) == 2:
        input_feed = {
            onnx_inputs[0].name: tensor_np,
            onnx_inputs[1].name: np.array([[w, h]], dtype=np.int64)
        }
    else:
        input_feed = {onnx_inputs[0].name: tensor_np}
    outputs = onnx_session.run(None, input_feed)
    if len(outputs) == 1:
        output = outputs[0]
        output_tensor = torch.from_numpy(output)
        if output_tensor.shape[-1] == 4 + len(idx_to_labels):
            boxes = output_tensor[0, :, :4]  
            logits = output_tensor[0, :, 4:]  
        elif output_tensor.shape[-1] > 4:
            boxes = output_tensor[0, :, :4]
            logits = output_tensor[0, :, 4:]
        else:
            raise ValueError(f"Unexpected ONNX output shape: {output_tensor.shape}")
        scores = torch.sigmoid(logits)
        max_scores, labels = torch.max(scores, dim=1)  
        outputs_tensor = {
            'pred_logits': logits,  
            'pred_boxes': boxes    
        }
        ori_size=torch.Tensor([w,h]).long().unsqueeze(0)
        result_ = postprocessor(outputs_tensor, ori_size)
        score_=result_[0]['scores']
        boxe_=result_[0]['boxes']
        label_=result_[0]['labels']
    elif len(outputs) == 2:
        outputs_tensor = {
            'pred_logits': torch.from_numpy(outputs[0]),
            'pred_boxes': torch.from_numpy(outputs[1])
        }
        ori_size=torch.Tensor([w,h]).long().unsqueeze(0)
        result_ = postprocessor(outputs_tensor, ori_size)
        score_=result_[0]['scores']
        boxe_=result_[0]['boxes']
        label_=result_[0]['labels']
    elif len(outputs) == 3:
        out0, out1, out2 = outputs[0], outputs[1], outputs[2]
        shapes = [(out0.shape, 0), (out1.shape, 1), (out2.shape, 2)]
        boxes_idx = None
        for shape, idx in shapes:
            if len(shape) >= 2 and shape[-1] == 4:
                boxes_idx = idx
                break
        if boxes_idx is None:
            boxes_idx = 0
        remaining = [0, 1, 2]
        remaining.remove(boxes_idx)
        out_rem1 = outputs[remaining[0]]
        out_rem2 = outputs[remaining[1]]
        if out_rem1.dtype in [np.float32, np.float64]:
            scores_idx = remaining[0]
            labels_idx = remaining[1]
        elif out_rem2.dtype in [np.float32, np.float64]:
            scores_idx = remaining[1]
            labels_idx = remaining[0]
        else:
            labels_idx = remaining[0]
            scores_idx = remaining[1]
        boxes_out = outputs[boxes_idx]
        scores_out = outputs[scores_idx]
        labels_out = outputs[labels_idx]
        if len(boxes_out.shape) == 3:
            boxes_out = boxes_out[0]
        if len(scores_out.shape) == 2:
            scores_out = scores_out[0]
        if len(labels_out.shape) == 2:
            labels_out = labels_out[0]
        score_ = torch.from_numpy(scores_out).float()
        boxe_ = torch.from_numpy(boxes_out).float()
        label_ = torch.from_numpy(labels_out).long()
    else:
        raise ValueError(f"Unexpected number of ONNX outputs: {len(outputs)}")
    selected_indices =score_ > box_thresh
    output={
    'labels': label_[selected_indices].to("cpu").numpy(),
    'boxes': boxe_[selected_indices].to("cpu").numpy(),
    'scores': score_[selected_indices].to("cpu").numpy()
    }
    center_coords=bbox2center(output['boxes'])
    output = {'bbox':         output["boxes"],
            'bbox_centers': center_coords,
            'scores':       output["scores"],
        'pred_classes': output["labels"]}
    output_a, output_b, output_c= ouptnp2abc(output,idx_to_labels)
    if debug:print("c,a,b>>>>>",len(output_c['pred_classes']),len(output_a['pred_classes']),len(output_b['pred_classes']))
    if len(output_a['pred_classes'])==0:
        file_path = f'{args.dataname}_Check_AboxIs0.txt'
        content = f'{image_path}@@{id_}---{image_path}\n'
        with open(file_path, 'a', encoding='utf-8') as f:
            f.write(content)
        continue 
    overlap_records = []
    to_remove = set()
    bond_boxes = output_b['bbox']
    bboxes = output_a['bbox'].copy()
    a_center = output_a['bbox_centers'].copy()
    scores = output_a['scores'].copy()
    pred_classes = output_a['pred_classes'].copy()
    to_remove = set()
    for i in range(len(bboxes)):
        for j in range(i + 1, len(bboxes)):
            x_min1, y_min1, x_max1, y_max1 = bboxes[i]
            x_min2, y_min2, x_max2, y_max2 = bboxes[j]
            x_min_inter = max(x_min1, x_min2)
            y_min_inter = max(y_min1, y_min2)
            x_max_inter = min(x_max1, x_max2)
            y_max_inter = min(y_max1, y_max2)
            inter_width = max(0, x_max_inter - x_min_inter)
            inter_height = max(0, y_max_inter - y_min_inter)
            inter_area = inter_width * inter_height
            area1 = (x_max1 - x_min1) * (y_max1 - y_min1)
            area2 = (x_max2 - x_min2) * (y_max2 - y_min2)
            union_area = area1 + area2 - inter_area
            iou = inter_area / union_area if union_area > 0 else 0
            score_i = scores[i] if scores[i] is not None else -1
            score_j = scores[j] if scores[j] is not None else -1
            if iou == 1:
                if score_i > score_j:
                    to_remove.add(j)
                else:
                    to_remove.add(i)
            elif iou>=0.8 and iou <1.0:
                if score_i > score_j:
                    to_remove.add(j)
                    if debug: print([i,j,score_i,score_j],iou,f"will remove j {j}, i-j {i,j}")
                else:
                    to_remove.add(i)
                    if debug: print([i,j,score_i,score_j],iou,f"will remove i {i}, i-j {i,j} ")
            elif iou > 0 and iou < 0.89 :
                if debug: print([i,j,score_i,score_j],iou,"<<<<<<111")
                if inter_area == area1 and area1 < area2:  
                    large_idx, small_idx = j, i
                elif inter_area == area2 and area2 < area1:  
                    large_idx, small_idx = i, j
                else:
                    if debug: print([i,j,score_i,score_j],iou,'OVERLAP without processed this version')
                    continue
                contains_bond = False
                for bond_bbox in bond_boxes:
                    if is_contained_in(bond_bbox, bboxes[large_idx]):
                        contains_bond = True
                        bboxes[large_idx] = adjust_bbox1(bboxes[large_idx], bboxes[small_idx], bond_bbox)
                        break
                if not contains_bond:
                    to_remove.add(small_idx)
            elif iou==0:
                pass
            else:
                print([i,j,score_i,score_j],iou,"<<<<<<222")
                print('what this case ???')   
    atom_bboxes = [bboxes[i] for i in range(len(bboxes)) if i not in to_remove]
    atom_scores = [scores[i] for i in range(len(scores)) if i not in to_remove]
    atom_centers = [a_center[i] for i in range(len(a_center)) if i not in to_remove]
    atom_classes = [pred_classes[i] for i in range(len(pred_classes)) if i not in to_remove]
    sorted_indices = sorted(range(len(atom_bboxes)), key=lambda i: (atom_bboxes[i][0], atom_bboxes[i][1]))
    atom_bboxes = [atom_bboxes[i] for i in sorted_indices]
    atom_scores = [atom_scores[i] for i in sorted_indices]
    atom_centers = [atom_centers[i] for i in sorted_indices]
    atom_classes = [atom_classes[i] for i in sorted_indices]
    print(len(atom_classes),'xxxxxxxx')
    bond_bbox = output_b['bbox'].copy()
    bond_scores = output_b['scores'].copy()
    bond_classes = output_b['pred_classes'].copy()
    if len(atom_bboxes)!=len(output_a['bbox']):
        if getacc:
            flogout2.write(f"fix ac json file---:  {file_base} \n")
    bonds = dict()
    b2aa = dict()
    singleAtomBond = dict()
    bondWithdirct = dict()
    _margin = 0
    bond_direction = dict()
    atom_centers_, atom_bbox_, original_to_subbox, subbox_to_original = preprocess_atom_boxes(atom_centers, atom_bboxes)
    tree_atom = KDTree(atom_centers_)
    if debug:
        print(f"KDTree built with {len(atom_centers_)} atom centers")
    for bi, (bbox, bond_type) in enumerate(zip(bond_bbox, bond_classes)):
        score = bond_scores[bi]
        if score is None:
            score = 1.0  
            bond_scores[bi] = score
        anchor_positions = (np.array(bbox) + [_margin, _margin, -_margin, -_margin]).reshape([2, -1])
        oposite_anchor_positions = anchor_positions.copy()
        oposite_anchor_positions[:, 1] = oposite_anchor_positions[:, 1][::-1]
        anchor_positions = np.concatenate([anchor_positions, oposite_anchor_positions])
        dists, neighbours = tree_atom.query(anchor_positions, k=1)
        if np.argmin((dists[0] + dists[1], dists[2] + dists[3])) == 0:
            begin_idx, end_idx = neighbours[:2]
        else:
            begin_idx, end_idx = neighbours[2:]
        atom1_idx = int(subbox_to_original[int(begin_idx)])
        atom2_idx = int(subbox_to_original[int(end_idx)])
        if atom1_idx == atom2_idx:
            if debug:
                print(f"singleAtomBond detected with bond id:{bi} atomIdx1 == atomIdx2 ::{[atom1_idx, atom2_idx]}")
            singleAtomBond[bi] = [atom1_idx]
        min_ai = min([atom1_idx, atom2_idx])
        max_ai = max([atom1_idx, atom2_idx])
        if debug:
            print(f"Bond {bi}: [{min_ai}, {max_ai}]")
        if bond_type in ['single', 'wdge', 'dash', '-', 'NONE', 'ENDUPRIGHT', 'BEGINWEDGE', 'BEGINDASH', 'ENDDOWNRIGHT']:
            bond_ = [min_ai, max_ai, 'SINGLE', score]
            if bond_type in ['wdge', 'dash', 'ENDUPRIGHT', 'BEGINWEDGE', 'BEGINDASH', 'ENDDOWNRIGHT']:
                bondWithdirct[bi] = [min_ai, max_ai, 'SINGLE', score, bond_type]
        elif bond_type == '=':
            bond_ = [min_ai, max_ai, 'DOUBLE', score]
        elif bond_type == '#':
            bond_ = [min_ai, max_ai, 'TRIPLE', score]
        elif bond_type == ':':
            bond_ = [min_ai, max_ai, 'AROMATIC', score]
        else:
            if debug:
                print(f"Unknown bond_type: {bond_type} for bond {bi} [{min_ai, max_ai}]")
            bond_ = [min_ai, max_ai, 'SINGLE', score]
        bonds[bi] = bond_
        b2aa[bi] = sorted([min_ai, max_ai])
    if debug:
        print(f"bonds {len(bonds)}, b2aa {len(b2aa)}, singleAtomBond {len(singleAtomBond)}, bondWithdirct {len(bondWithdirct)}")
    a2b=dict()
    isolated_a=set()
    aa2b_d2=dict()
    for k,v in b2aa.items():
        vt=(v[0],v[1])
        if vt in aa2b_d2:
            aa2b_d2[vt].append(k)
        else:
            aa2b_d2[vt]=[k]
        for a in set(v):
            if a not in a2b.keys():
                a2b[a]=[k]
            else:
                a2b[a].append(k)
    a2neib = {}
    for atom, bns in a2b.items():
        neighbors = set()  
        for bond in bns:
            atom_pair = b2aa[bond]  
            nei={ai for ai in atom_pair if ai !=atom }
            neighbors.update(nei)
        a2neib[atom] = sorted(list(neighbors))  
    isolated_a=set()
    for ai, a_lab in enumerate(atom_classes):
        if ai not in a2b.keys():
            isolated_a.add(ai)
    if debug:print("detected possible isolated atom:", isolated_a)
    repeate_bonds={k:v for k,v in aa2b_d2.items() if len(v)>=2 }
    if debug:print(f"repeat bond box ids {repeate_bonds}")
    if len(isolated_a)>0:
        isolated_a2del=[]
        bond_sizes = []
        for bbox in bond_bbox:
            width = bbox[2] - bbox[0]
            height = bbox[3] - bbox[1]
            size = min(width, height)  
            bond_sizes.append(size)
        min_bond_size = min(bond_sizes) if bond_sizes else 10.0  
        if debug:print("min_bond_size ",min_bond_size, 10)
        new_bond_idx = len(bond_bbox)
        isolated_aFound=[]
        singleAtomBond_fixed=[]
        for iso_atom in isolated_a:
            iso_box = atom_bboxes[iso_atom]
            for bi,atom_idx_list in singleAtomBond.items():
                bond_box = bond_bbox[bi]
                atom1_idx = atom_idx_list[0]
                bond_vertices = get_corners(bond_box)
                atom1_center = atom_centers[atom1_idx]
                distances = [np.linalg.norm(np.array(atom1_center) - v) for v in bond_vertices]
                closest_indices = np.argsort(distances)[:2] 
                connected_vertices = bond_vertices[closest_indices]
                unconnected_vertices = bond_vertices[[i for i in range(4) if i not in closest_indices]]
                exclude_=[atom1_idx]+a2neib[atom1_idx]
                print(f'exclude this atom itself :: {exclude_},and its neiboughs {a2neib[atom1_idx]}')
                atom2_idx_, dist2 = find_nearest_atom(unconnected_vertices, atom_bboxes, exclude_idx=exclude_)
                if iso_atom == atom2_idx_:
                    if atom2_idx_< atom1_idx:
                        k=[atom2_idx_, atom1_idx]
                    else:
                        k=[atom1_idx, atom2_idx_]
                    if atom2_idx_ not in a2neib[atom1_idx]:
                        b2aa[bi]=k
                        bonds[bi][0]=k[0]
                        bonds[bi][1]=k[1]
                        a2b.setdefault(iso_atom, []).append(bi)
                        if debug: print(f'@@isolated_a fix the SingleAtomBond {bi} as bond:{bonds[bi]} !!')
                        singleAtomBond_fixed.append(bi)
                        isolated_aFound.append(atom2_idx_)
            if len(repeate_bonds)>0:
                at2b_dist=dict()
                iso_box_vertices = get_corners(iso_box)
                iso_atom_center = atom_centers[iso_atom]
                bond_box_idx_, bond_box_dist = find_nearest_atom(iso_box_vertices, bond_bbox, exclude_idx=[])
                for a1a2,bis in repeate_bonds.items():
                    for bi in bis:
                        if bi ==bond_box_idx_:
                            bond_box = bond_bbox[bi]
                            bond_vertices = get_corners(bond_box)
                            a1_,a2_=a1a2
                            a1_atombox= atom_bboxes[a1_]
                            a2_atombox= atom_bboxes[a2_]
                            a1_flag= boxes_overlap(a1_atombox, bond_box)
                            a2_flag= boxes_overlap(a2_atombox, bond_box)
                            if a1_flag: 
                                atom1_idx_=a1_
                                dist1=0
                            elif a2_flag: 
                                atom1_idx_=a2_
                                dist1=0
                            else:
                                distances = [np.linalg.norm(np.array(iso_atom_center) - v) for v in bond_vertices]
                                closest_indices2 = np.argsort(distances)[:2] 
                                connected_vertices2 = bond_vertices[closest_indices2]
                                connected_vertices1 = bond_vertices[[i for i in range(4) if i not in closest_indices2]]
                                atom1_idx_, dist1 = find_nearest_atom(connected_vertices1, atom_bboxes, exclude_idx=[iso_atom])
                            if debug:print("a1_flag,a2_flag,atom1_idx_, iso_atom",[a1_flag,a2_flag,atom1_idx_,iso_atom])
                            min_ai=min([atom1_idx_,iso_atom])
                            max_ai=max([atom1_idx_,iso_atom])
                            k=(min_ai,max_ai)
                            print(k,'repeate',bi)
                            if k not in at2b_dist:
                                at2b_dist[k]=[bi,a1a2,dist1]
                            else:
                                if dist1< at2b_dist[k][1]:
                                    at2b_dist[k]=[bi,a1a2,dist1]
                            if debug:print(f"repate bond box id: {bi} fixed with {at2b_dist}")
                            isolated_aFound.append(iso_atom)
                            isolated_a2del.append(iso_atom)
                            b2aa[bi] = [min_ai,max_ai]
                            a2b.setdefault(iso_atom, []).append(bi)
                            bonds[bi][0]=k[0]
                            bonds[bi][1]=k[1]
                            if bi in bondWithdirct:
                                bondWithdirct[bi][0]=k[0]
                                bondWithdirct[bi][1]=k[1]
        isolated_a=[ ai for ai in isolated_a if ai not in isolated_aFound]
        singleAtomBond={bi:aili for bi,aili in singleAtomBond.items() if bi not in singleAtomBond_fixed}
        for iso_atom in isolated_a:
            iso_box = atom_bboxes[iso_atom]
            for other_idx, other_box in enumerate(atom_bboxes):
                if other_idx == iso_atom                    or (atom_classes[other_idx] in ['other',"*"] and atom_classes[iso_atom] in ['other',"*"]):
                    continue
                min_ai=min([iso_atom,other_idx])
                max_ai=max([iso_atom,other_idx])
                if boxes_overlap(iso_box, other_box):
                    new_bbox = [
                        min(iso_box[0], other_box[0]),
                        min(iso_box[1], other_box[1]),
                        max(iso_box[2], other_box[2]),
                        max(iso_box[3], other_box[3])
                    ]
                    bond_bbox.append(new_bbox)
                    bond_classes.append('single')
                    bond_scores.append(1.0)
                    b2aa[new_bond_idx] = [iso_atom, other_idx]
                    a2b.setdefault(iso_atom, []).append(new_bond_idx)
                    a2b.setdefault(other_idx, []).append(new_bond_idx)
                    isolated_a2del.append(iso_atom)
                    new_bond_idx += 1
                    bond_=[min_ai, max_ai, 'SINGLE', 1.0]
                    last_=len(bonds)
                    bonds[last_] = bond_
                    if debug:
                        print(f"添加键 {new_bond_idx-1} 连接原子 {iso_atom} 和 {other_idx},as isoated box overlap with it ")
                else:
                    min_dist = float('inf')
                    closest_atom = None
                    dist = min_corner_distance(iso_box, other_box)
                    if dist < min_dist:
                        min_dist = dist
                        closest_atom = other_idx
                    if min_dist < min_bond_size:
                        new_bbox = [
                            min(iso_box[0], atom_bboxes[closest_atom][0]),
                            min(iso_box[1], atom_bboxes[closest_atom][1]),
                            max(iso_box[2], atom_bboxes[closest_atom][2]),
                            max(iso_box[3], atom_bboxes[closest_atom][3])
                        ]
                        bond_bbox.append(new_bbox)
                        bond_classes.append('single')
                        bond_scores.append(1.0)
                        b2aa[new_bond_idx] = [iso_atom, closest_atom]
                        a2b.setdefault(iso_atom, []).append(new_bond_idx)
                        a2b.setdefault(closest_atom, []).append(new_bond_idx)
                        isolated_a2del.append(iso_atom)
                        new_bond_idx += 1
                        if debug:
                            print(f"添加键 {new_bond_idx-1} 连接原子 {iso_atom} 和 {closest_atom} (距离 {min_dist})")
                        bond_=[min_ai, max_ai, 'SINGLE', 1.0]
                        last_=len(bonds)
                        bonds[last_] = bond_
        if debug:
            print('isolated_a2del and isolated_a number',len(isolated_a2del),len(isolated_a))
            print('isolated_a ',isolated_a)
            print('isolated_a2del ',isolated_a2del)
    a2b = dict(sorted(a2b.items()))
    if len(singleAtomBond) > 0:
        skip_none_mol = False
        a2neib = {}
        for atom, bns in a2b.items():
            neighbors = set()  
            for bond in bns:
                atom_pair = b2aa[bond]  
                nei={ai for ai in atom_pair if ai !=atom }
                neighbors.update(nei)
            a2neib[atom] = sorted(list(neighbors))  
        c_bboxes = [bbox for bbox, cls in zip(output_a['bbox'], output_a['pred_classes']) if cls == 'C']
        if not c_bboxes:
            print("Warning: No 'C' atoms found, using smallest bbox in output_a instead.")
            all_bboxes = output_a['bbox']
            if not all_bboxes:
                raise ValueError("No bboxes found in output_a at all.")            
            smallest_bbox = min(all_bboxes, key=bbox_area)
            c_bboxes = [smallest_bbox]    
        min_width = min([bbox[2] - bbox[0] for bbox in c_bboxes])
        min_height = min([bbox[3] - bbox[1] for bbox in c_bboxes])
        for bi, atom_idx_list in singleAtomBond.items():
            bond_box = bond_bbox[bi]
            atom1_idx = atom_idx_list[0]
            bond_vertices = get_corners(bond_box)
            atom1_center = atom_centers[atom1_idx]
            distances = [np.linalg.norm(np.array(atom1_center) - v) for v in bond_vertices]
            closest_indices = np.argsort(distances)[:2] 
            connected_vertices = bond_vertices[closest_indices]
            unconnected_vertices = bond_vertices[[i for i in range(4) if i not in closest_indices]]
            exclude_=[atom1_idx]
            print(f'exclude this atom itself :: {exclude_},and its neiboughs {a2neib[atom1_idx]}')
            atom2_idx_, dist2 = find_nearest_atom(unconnected_vertices, atom_bboxes, exclude_idx=exclude_)
            atom1_corners = get_corners(atom_bboxes[atom1_idx])
            atom2_1_idx, dist2_1 = find_nearest_atom(atom1_corners, atom_bboxes, exclude_idx=exclude_)
            if debug:print("atom2_idx_ , atom2_1_idx,atom1_idx:",atom2_idx_, atom2_1_idx,atom1_idx)
            if atom2_idx_ is None or atom2_1_idx is None:
                reason = 'singleAtomBond_has_no_second_atom'
                print(
                    f"skip none-molecular image: {image_path} | reason: {reason} | "
                    f"atom2_idx_={atom2_idx_}, atom2_1_idx={atom2_1_idx}, atom1_idx={atom1_idx}"
                )
                none_mol_images.append({
                    'id': id_,
                    'file_name': os.path.basename(image_path),
                    'file_base': file_base,
                    'file_path': image_path,
                    'SMILESori': SMILESori,
                    'reason': reason,
                    'atom1_idx': atom1_idx,
                    'atom2_idx': atom2_idx_,
                    'atom2_1_idx': atom2_1_idx,
                })
                skip_none_mol = True
                break
            if atom2_idx_< atom1_idx:
                k=[atom2_idx_, atom1_idx]
            else:
                k=[atom1_idx, atom2_idx_]
            if atom2_idx_ == atom2_1_idx :
                if atom2_idx_ not in a2neib[atom1_idx]:
                    if debug: print('add new bond with existed atom')
                    b2aa[bi]=k
                    bonds[bi][0]=k[0]
                    bonds[bi][1]=k[1]
                else:
                    new_center=np.mean(unconnected_vertices, axis=0)
                    new_bbox = [
                        new_center[0] - min_width / 2,
                        new_center[1] - min_height / 2,
                        new_center[0] + min_width / 2,
                        new_center[1] + min_height / 2
                    ]
                    if debug: print('new atom box adding as C')
                    atom_bboxes.append(new_bbox)
                    atom_centers.append(new_center.tolist())
                    atom_scores.append(bond_scores[bi])  
                    atom_classes.append('C')
                    atom2_idx_= len(atom_classes)-1
                    k=[atom1_idx, atom2_idx_]
                    bonds[bi][1]=atom2_idx_
                    b2aa[bi][1]=atom2_idx_
            else:     
                if atom2_idx_ not in a2neib[atom1_idx]:
                    if debug: print(f'atom2_idx_ != atom2_1_idx| {atom2_idx_} != {atom2_1_idx} @add new bond with existed atom')
                    b2aa[bi]=k
                    bonds[bi][0]=k[0]
                    bonds[bi][1]=k[1]
                else:
                    new_center=np.mean(unconnected_vertices, axis=0)
                    new_bbox = [
                        new_center[0] - min_width / 2,
                        new_center[1] - min_height / 2,
                        new_center[0] + min_width / 2,
                        new_center[1] + min_height / 2
                    ]
                    atom_bboxes.append(new_bbox)
                    atom_centers.append(new_center.tolist())
                    atom_scores.append(bond_scores[bi])  
                    atom_classes.append('C')
                    atom2_idx_= len(atom_classes)-1
                    k=[atom1_idx, atom2_idx_]
                    bonds[bi][1]=atom2_idx_
                    b2aa[bi][1]=atom2_idx_
                    if debug: print(f'atom2_idx_ != atom2_1_idx@new atom box {atom2_idx_}adding as C, with bond {bi} a1a2 {k}')
            if bi in bondWithdirct.keys():
                bondWithdirct[bi][0]=k[0]
                bondWithdirct[bi][1]=k[1]
        if skip_none_mol:
            continue
    if debug:print(f"before del bonds  {len(bond_bbox)}")
    aa2b=dict()
    for bi, aa in b2aa.items():
        min_ai=min(aa)
        max_ai=max(aa)
        if bond_scores[bi] is None:
            bond_scores[bi]=1.0
        score_=bond_scores[bi]
        bond_type=bond_classes[bi]
        if bond_type in ['single','wdge','dash', '-', 'NONE', 'ENDUPRIGHT', 'BEGINWEDGE', 'BEGINDASH', 'ENDDOWNRIGHT']:
            bond_ = [min_ai, max_ai, 'SINGLE', score]
            if bond_type in ['wdge','dash','ENDUPRIGHT', 'BEGINWEDGE', 'BEGINDASH', 'ENDDOWNRIGHT']:
                bondWithdirct[bi]=[min_ai, max_ai,'SINGLE', score, bond_type]
        elif bond_type == '=':
            bond_ = [min_ai, max_ai, 'DOUBLE', score]
        elif bond_type == '#':
            bond_ = [min_ai, max_ai, 'TRIPLE', score]
        elif bond_type == ':':
            bond_ = [min_ai, max_ai, 'AROMATIC', score]
        else:
            print(f"what case here !!! with bond_type: {bond_type} || {[bi,min_ai, max_ai]}")
            bond_=[min_ai, max_ai, 'SINGLE', score]
        if (min_ai, max_ai) not in aa2b.keys() or aa2b[(min_ai, max_ai)][-2]<score_:
            aa2b[(min_ai, max_ai)]=[bi,score_,bond_[-2]]
    if len(aa2b)!=len(b2aa):
        new_bi_map = {}  
        new_bonds = {}
        new_aa2b = {}
        new_b2aa = {}
        new_bondWithdirct = {}
        new_singleAtomBond = {}
        for new_bi, ((min_ai, max_ai), (old_bi, score, bond_type)) in enumerate(
            sorted(aa2b.items(), key=lambda x: x[1][1], reverse=True)  
        ):
            new_bi_map[old_bi] = new_bi
            new_bonds[new_bi] = [min_ai, max_ai, bond_type, score]
            new_aa2b[(min_ai, max_ai)] = [new_bi, score, bond_type]
            new_b2aa[new_bi] = [min_ai, max_ai]
        for old_bi, bond_info in bondWithdirct.items():
            if old_bi in new_bi_map:
                new_bi = new_bi_map[old_bi]
                new_bondWithdirct[new_bi] = bond_info
        for old_bi, bond_info in singleAtomBond.items():
            if old_bi in new_bi_map:
                new_bi = new_bi_map[old_bi]
                new_singleAtomBond[new_bi] = bond_info
        bonds = new_bonds
        aa2b = new_aa2b
        b2aa = new_b2aa
        bondWithdirct = new_bondWithdirct
        singleAtomBond = new_singleAtomBond
        if debug: print(f"去重完成: bonds={len(bonds)}, aa2b={len(aa2b)}, b2aa={len(b2aa)}, bondWithdirct={len(bondWithdirct)}")
        old_bns=max(new_bi_map.keys())
        to_remove_bonds=set()
        for i in range(old_bns):
            if i not in new_bi_map.keys():
                to_remove_bonds.add(i)
        print(to_remove_bonds)
        bond_scores = [bond_scores[i] for i in range(len(bond_scores)) if i not in to_remove_bonds]
        bond_classes = [bond_classes[i] for i in range(len(bond_classes)) if i not in to_remove_bonds]
        bond_bbox = [bond_bbox[i] for i in range(len(bond_bbox)) if i not in to_remove_bonds]
        bond_center = [[ (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2 ] for bbox in bond_bbox]
    a2b=dict()
    isolated_a=set()
    for k,v in b2aa.items():
        for a in v:
            if a not in a2b.keys():
                a2b[a]=[k]
            else:
                a2b[a].append(k)
    for ai, a_lab in enumerate(atom_classes):
        if ai not in a2b.keys():
            isolated_a.add(ai)
    a2b = dict(sorted(a2b.items()))
    a2neib = {}
    for atom, bns in a2b.items():
        neighbors = set()  
        for bond in bns:
            atom_pair = b2aa[bond]  
            nei={ai for ai in atom_pair if ai !=atom }
            neighbors.update(nei)
        a2neib[atom] = sorted(list(neighbors))  
    debug2=False
    if debug2:
        print("\nBonds:")
        for bi, bond_info in bonds.items():
            print(f"Bond {bi}: {bond_info}")
        print("\nSingle Atom Bonds:")
        for bi, atom_idx in singleAtomBond.items():
            print(f"Bond {bi}: {atom_idx}")
        print("Atom to Bonds box idx maping:")
        for ai, bond_ids in a2b.items():
            print(f"a2b-id {ai}: {bond_ids}")
        print(f"isolated_ atom box:: {isolated_a}")
        print(f"b2aa::{b2aa}")
        print("a2neib:")
        for atom, neighbors in a2neib.items():
            print(f"Atom {atom}: {neighbors}")
    other2ppsocr = True
    ocr_ai2lab = dict()
    ocr_bbs = dict()
    scale_crop = False
    ocr_ai2lab_ori=dict()
    ocr_ai2lab_sca=dict()
    if other2ppsocr:
        elements = ['S', 'N', 'P', 'C', 'O']
        keys = [f"{e}{suffix}" for e in elements for suffix in ['R"', "R'", "R", "*"]]
        replacement_map = {key: f'{key[0]}*' for key in keys}
        if da=='staker':
            _margin=2
        else:
            _margin=0
        for i, atc in enumerate(atom_classes):
            if 'other' == atc:  
                orig_text = ""
                orig_score = 0
                scaled_text = ""
                scaled_score = 0
                abox_orig = np.array(atom_bboxes[i]) + np.array([-_margin, -_margin,_margin, _margin])
                cropped_img_orig = img_ori.crop(abox_orig)
                image_npocr_orig = np.array(cropped_img_orig)
                result_ocr_orig = None
                if image_npocr_orig.size > 0 and image_npocr_orig.shape[0] > 0 and image_npocr_orig.shape[1] > 0:
                    result_ocr_orig = ocr.ocr(image_npocr_orig, det=False)
                if result_ocr_orig:
                    orig_text, orig_score = extract_ocr_text_score(result_ocr_orig)
                    if debug: print(f'oriCrop:\t {orig_text}   {orig_score}')
                    orig_text = normalize_ocr_text(orig_text, replacement_map)
                    ocr_ai2lab_ori[i]=[orig_text,orig_score]
                abox_scaled = np.array(atom_bboxes[i]) * np.array([scale_x, scale_y, scale_x, scale_y]) +  np.array([-_margin, -_margin,_margin, _margin])
                cropped_img_scaled = img_ori_1k.crop(abox_scaled)
                image_npocr_scaled = np.array(cropped_img_scaled)
                result_ocr_scaled = None
                if image_npocr_scaled.size > 0 and image_npocr_scaled.shape[0] > 0 and image_npocr_scaled.shape[1] > 0:
                    result_ocr_scaled = ocr.ocr(image_npocr_scaled, det=False)
                if result_ocr_scaled:
                    scaled_text, scaled_score = extract_ocr_text_score(result_ocr_scaled)
                    if debug:  print(f'scaled:\t {scaled_text}   {scaled_score}')
                    scaled_text = normalize_ocr_text(scaled_text, replacement_map)
                    ocr_ai2lab_sca[i]=[scaled_text,scaled_score]
                final_text, final_score, final_crop = select_chem_expression(
                    orig_text, orig_score, scaled_text, scaled_score, cropped_img_orig, cropped_img_scaled
                )
                if orig_text=='NO2' or scaled_text=='NO2':
                    final_text='NO2'
                elif orig_text=='SO2' or scaled_text=='SO2':
                    final_text='SO2'
                if final_text:
                    ocr_ai2lab[i] = [final_text, final_score]
                    ocr_bbs[i] = final_crop
                    atom_classes[i] = final_text
        if debug:
            print("ori",ocr_ai2lab_ori)
            print("sca",ocr_ai2lab_sca)
        print(ocr_ai2lab)
    if len(ocr_bbs)>0:
        if debug:print(f'numbs of ocr {len(ocr_bbs)} crop_ images')
    giveup_isolateds=dict()
    if len(isolated_a):
        for iso_atom in isolated_a:
            atom1_corners = get_corners(atom_bboxes[iso_atom])
            atom2_1_idx, dist2_1 = find_nearest_atom(atom1_corners, atom_bboxes, exclude_idx=[iso_atom])
            atom1_lab=atom_classes[iso_atom]
            atom2_lab=atom_classes[atom2_1_idx]
            if atom1_lab in ['Ph3Br','Ph3Br-']:
                if iso_atom not in giveup_isolateds.keys():
                    giveup_isolateds[iso_atom]=[atom1_lab]
                else:
                    giveup_isolateds[iso_atom].append(atom1_lab)
                if atom2_lab in ['P','P+']:
                    atom2_lab='P+Ph3Br-'
                elif atom2_lab in ['N','N+']:
                    atom2_lab='N+Ph3Br-'
            atom_classes[atom2_1_idx]=atom2_lab 
    if debug:
        print(f"giveup_isolateds {giveup_isolateds}")
        print(len(atom_classes),len(bond_classes),'<<<<<<<<<<<')
    rwmol_ = Chem.RWMol()
    boxi2ai = {}  
    placeholder_atoms=dict()
    J=0
    for i, (bbox, a) in enumerate(zip(atom_bboxes, atom_classes)):
        a2labl=False
        a=replace_cg_notation(a)
        if a in ['H', 'C', 'O', 'N', 'Cl', 'Br', 'S', 'F', 'B', 'I', 'P', 'Si']:
            ad = Chem.Atom(a)
        elif a in ELEMENTS:
            ad = Chem.Atom(a)
        elif a in ABBREVIATIONS :
            ad = Chem.Atom("*")
            placeholder_atoms[i] = a 
            a2labl=True
        else:
            if  N_C_H_expand(a):
                ad = Chem.Atom("*")
                placeholder_atoms[i] = a 
                a2labl=True
            elif C_H_expand(a):
                ad = Chem.Atom("*")
                placeholder_atoms[i] = a 
                a2labl=True
            elif C_H_expand2(a):
                        ad = Chem.Atom("*")
                        placeholder_atoms[i] = a 
                        a2labl=True
            elif  formula_regex(a):
                ad = Chem.Atom("*")
                placeholder_atoms[i] = a 
                a2labl=True
            else:
                ad = Chem.Atom("*")
                if a not in ['*',"other"]:
                    a2labl=True
        rwmol_.AddAtom(ad)
        boxi2ai[J] = rwmol_.GetNumAtoms() - 1
        if a2labl: rwmol_.GetAtomWithIdx(J).SetProp("atomLabel", f"{a}")
        J+=1
    charges_classes= output_c['pred_classes']
    charges_centers= output_c['bbox_centers']
    charges_scores= output_c['scores']
    charges_bbox=  output_c['bbox']
    a2c=dict()
    c2a=dict()
    if len(charges_classes) > 0:
        kdt = cKDTree(atom_centers)
        c2a = {}  
        used_atoms = set()  
        for i, charge_box in enumerate(charges_bbox):
            charge_value = parse_charge(charges_classes[i])
            overlapped_atoms = []
            for ai, atom_box in enumerate(atom_bboxes):
                if boxes_overlap(charge_box, atom_box):
                    overlapped_atoms.append(ai)
            if overlapped_atoms:
                for ai in overlapped_atoms:
                    if ai not in used_atoms:
                        c2a[i] = ai
                        used_atoms.add(ai)
                        break
            else:
                x, y = charges_centers[i]
                dist_kdt, ai_kdt = kdt.query([x, y], k=1)
                min_dist = float('inf')
                ai_corner = None
                for ai, atom_box in enumerate(atom_bboxes):
                    dist = min_corner_distance(charge_box, atom_box)
                    if dist < min_dist:
                        min_dist = dist
                        ai_corner = ai
                if ai_kdt == ai_corner and ai_kdt not in used_atoms:
                    c2a[i] = ai_kdt
                    used_atoms.add(ai_kdt)
                else:
                    if charge_value != 0:
                        symbol_kdt =atom_classes[ai_kdt]
                        symbol_corner =atom_classes[ai_corner]
                        if symbol_kdt == 'C' and symbol_corner != 'C' and ai_corner not in used_atoms:
                            c2a[i] = ai_corner
                            used_atoms.add(ai_corner)
                        elif symbol_corner == 'C' and symbol_kdt != 'C' and ai_kdt not in used_atoms:
                            c2a[i] = ai_kdt
                            used_atoms.add(ai_kdt)
                        else:
                            if ai_kdt not in used_atoms:
                                c2a[i] = ai_kdt
                                used_atoms.add(ai_kdt)
                            elif ai_corner not in used_atoms:
                                c2a[i] = ai_corner
                                used_atoms.add(ai_corner)
        a2c={v:k for k,v in c2a.items()}
        for k,v in a2c.items():
            fc=int(charges_classes[v])
            rwmol_.GetAtomWithIdx(k).SetFormalCharge(fc)
            if atom_classes[k] in ['COO','CO2']:
                if fc==-1:
                    atom_classes[k]=f"{atom_classes[k]}-"
                    placeholder_atoms[k]=atom_classes[k]
                    atom = rwmol_.GetAtomWithIdx(k)
                    atom.SetProp("atomLabel",placeholder_atoms[k])
                elif fc==1:
                    atom_classes[k]=f"{atom_classes[k]}+"
                    placeholder_atoms[k]=atom_classes[k]
                    atom = rwmol_.GetAtomWithIdx(k)
                    atom.SetProp("atomLabel",placeholder_atoms[k])
                else:
                    print(f"charge adding {fc} @ {atom_classes[v]}")
        print(f'placeholder_atoms {placeholder_atoms}')
    for bi, bond in bonds.items():
        atom1_idx, atom2_idx, bond_type, score = bond
        if atom1_idx ==atom2_idx:print(f"self bond should be avoid or del on previous process!!")
        if bond_type == 'SINGLE':
            rwmol_.AddBond(atom1_idx, atom2_idx, Chem.BondType.SINGLE)
        elif bond_type == 'DOUBLE':
            rwmol_.AddBond(atom1_idx, atom2_idx, Chem.BondType.DOUBLE)
        elif bond_type == 'TRIPLE':
            rwmol_.AddBond(atom1_idx, atom2_idx, Chem.BondType.TRIPLE)
        elif bond_type == 'AROMATIC':
            rwmol_.AddBond(atom1_idx, atom2_idx, Chem.BondType.AROMATIC)
        else:
            print(f"Unknown bond type: {bond_type}")
    if debug:    print(f"all a2b b2a a2c c2a done, start mol built done")
    if len(bondWithdirct)>0:
        print(f"set bond direction for mollecule ")
    skeleton_smi = Chem.MolToSmiles(rwmol_)
    coords = [(x,-y,0) for x,y in atom_centers]
    coords = tuple(coords)
    coords = tuple(tuple(num / 100 for num in sub_tuple) for sub_tuple in coords)
    mol2D = rwmol_.GetMol()
    mol2D.RemoveAllConformers()
    conf = Chem.Conformer(mol2D.GetNumAtoms())
    conf.Set3D(True)
    for i, (x, y, z) in enumerate(coords):
        conf.SetAtomPosition(i, (x, y, z))
    mol2D.AddConformer(conf)
    try:
        Chem.SanitizeMol(mol2D)
        Chem.AssignStereochemistryFrom3D(mol2D)
        mol_rebuit2d=Chem.RWMol(mol2D) 
    except Exception as e:
        print(e)
        print('before expanding!!! try to sanizemol and assign stereo')
        mol_rebuit2d=Chem.RWMol(rwmol_) 
    if len(giveup_isolateds)>0:
        for atom in mol_rebuit2d.GetAtoms():
            atom.SetProp('old_index', str(atom.GetIdx()))
        for ai in sorted(giveup_isolateds.keys(), reverse=True):
            mol_rebuit2d.RemoveAtom(ai)
            print(f"atom {ai} label {giveup_isolateds[ai]} removed")
        old_to_new = {}
        for atom in mol_rebuit2d.GetAtoms():
            old_idx = int(atom.GetProp('old_index'))
            new_idx = atom.GetIdx()
            old_to_new[old_idx] = new_idx
        if len(placeholder_atoms)>0:
            placeholder_atoms2=dict()
            for k,v in placeholder_atoms.items():
                placeholder_atoms2[old_to_new[k]]=v
            placeholder_atoms=placeholder_atoms2    
    try:
        SMILESpre = Chem.MolToSmiles(mol_rebuit2d)
    except Exception as e:
        print(f"Error during SMILES generation: {e}")
        SMILESpre = Chem.MolToSmiles(mol_rebuit2d, canonical=False)
    if len(placeholder_atoms)>0:
        mol_expan=copy.deepcopy(mol_rebuit2d)
        if debug: print(f'MOL will be expanded with {placeholder_atoms} !!')
        wdbs=[]
        bond_dirs_rev={v:k for k,v in bond_dirs.items()}
        for b in mol_expan.GetBonds():
            bd=b.GetBondDir()
            bt=b.GetBondType()
            if bd ==bond_dirs['BEGINDASH'] or  bd==bond_dirs['BEGINWEDGE']:
                a1, a2 = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
                wdbs.append([a1,a2,bt,bond_dirs_rev[bd]])
        expandStero_smi1,molexp= molExpanding(mol_expan,placeholder_atoms,wdbs,bond_dirs)
        molexp=remove_bond_directions_if_no_chiral(molexp)
        try:
            Chem.SanitizeMol(molexp)
            expandStero_smi=Chem.MolToSmiles(molexp)
        except Exception as e:
            print(f"Error during sanitization: {e}")
            expandStero_smi = expandStero_smi1
        expandStero_smi=remove_SP(expandStero_smi)
    else:
        molexp=mol_rebuit2d
        expandStero_smi=SMILESpre 
    new_row = {'file_name':image_path, "SMILESori":SMILESori,
                    'SMILESpre':SMILESpre,
                    'SMILESexp':expandStero_smi, 
                    }
    smiles_data = smiles_data._append(new_row, ignore_index=True)
    if getacc:
        has_ref=True
        if (type(SMILESori)!=type('a')):
            sameWithOutStero=True
            sameWithOutStero_exp=True
            has_ref=False
        else:
            sameWithOutStero= comparing_smiles(new_row,SMILESpre)
            sameWithOutStero_exp= comparing_smiles(new_row,expandStero_smi)
            expandStero_smi= select_longest_smiles(expandStero_smi)
            SMILESori= select_longest_smiles(SMILESori)
            logest_flag=comparing_smiles2(expandStero_smi,SMILESori)
            expandStero_fixed_smiles, warnings=check_and_fix_valence(expandStero_smi)
            logest_flagHfix= comparing_smiles2(expandStero_fixed_smiles,SMILESori)
            if logest_flagHfix:
                expandStero_smi=logest_flagHfix
            try:
                expandStero_smi_keku_canon= Chem.MolToSmiles(molexp,kekuleSmiles=True)
            except Exception as e:
                print('kekuleSmiles erros!!!!!\n ',e)
                expandStero_smi_keku_canon=''
            if logest_flag or logest_flagHfix:
                print('>>>>>>same smiles')
            else:
                print(f"expand_keku, expand,SMILESori @@@ id_ {id_}||row@{id_+2} ")
                print(expandStero_smi_keku_canon)
                print(expandStero_smi)
                print(SMILESori)
                print('@@@@@@@@@ not same smiles, ')
        if ac_b or rt_out or ac_b_smilesnotsame_writJson:
            ac_dict={  "version": "2.5.4",
                        "flags": {},
                    }
            b_dict={  "version": "2.5.4",
                        "flags": {},
                    }
            bboxes_ac=atom_bboxes+charges_bbox
            classes_ac=atom_classes+charges_classes
            shapes_ac= bbox2shapes(bboxes_ac, classes_ac, lab2idx)
            shapes_b= bbox2shapes(bond_bbox, bond_classes, lab2idx)
            ac_dict["shapes"]=shapes_ac
            b_dict["shapes"]=shapes_b
            b_dict["imagePath"]=f"{file_base}.png"
            b_dict["imageData"]=None
            b_dict["imageHeight"]=int(h_ori)
            b_dict["imageWidth"]=int(w_ori)
            b_dict["description"]=f"{file_base}.png\nid_={id_}"
            ac_dict["imagePath"]=f"{file_base}.png"
            ac_dict["imageData"]=None
            ac_dict["imageHeight"]=int(h_ori)
            ac_dict["imageWidth"]=int(w_ori)
            ac_dict["description"]=f"{file_base}.png\nid_={id_}"
        if sameWithOutStero or sameWithOutStero_exp or logest_flag or logest_flagHfix:
            mysum += 1
            if rt_out and has_ref:
                b_datadir=  os.path.join(view_dirb3, f"{file_base}.json")
                ac_datadir=  os.path.join(view_dirac3, f"{file_base}.json")
                b_datadir_img=  os.path.join(view_dirb3, f"{file_base}.png")
                ac_datadir_img=  os.path.join(view_dirac3, f"{file_base}.png")
                abs_image_path = os.path.abspath(image_path)
                abs_ac_datadir_img = os.path.abspath(ac_datadir_img)
                if abs_image_path != abs_ac_datadir_img:
                    shutil.copy(image_path, ac_datadir_img)
                    shutil.copy(image_path, b_datadir_img)
                else:
                    pass 
                with open(ac_datadir,"w") as wf1:
                        json.dump(ac_dict, wf1, indent=2)
                with open(b_datadir,"w") as wf2:
                        json.dump(b_dict, wf2, indent=2)
                ac_datadir_imgD=  os.path.join(view_dirac2, f"{file_base}.png")                        
                os.remove(ac_datadir_imgD)
            if ac_b and has_ref:
                b_datadir=  os.path.join(view_dirb, f"{file_base}.json")
                ac_datadir=  os.path.join(view_dirac, f"{file_base}.json")
                b_datadir_img=  os.path.join(view_dirb, f"{file_base}.png")
                ac_datadir_img=  os.path.join(view_dirac, f"{file_base}.png")
                abs_image_path = os.path.abspath(image_path)
                abs_ac_datadir_img = os.path.abspath(ac_datadir_img)
                if abs_image_path != abs_ac_datadir_img:
                    shutil.copy(image_path, ac_datadir_img)
                    shutil.copy(image_path, b_datadir_img)
                else:
                    pass 
                with open(ac_datadir,"w") as wf1:
                        json.dump(ac_dict, wf1, indent=2)
                with open(b_datadir,"w") as wf2:
                        json.dump(b_dict, wf2, indent=2)
        else:
            print(f"smiles problems\n{SMILESori}\n{SMILESpre}\n{image_path}")
            failed_fb.append(file_base)
            failed.append([SMILESori,SMILESpre,expandStero_smi,id_, image_path])
            SMILESori, success= rdkit_canonicalize_smiles(SMILESori)
            if success:
                molOri=Chem.MolFromSmiles(SMILESori)
                atom_num=len(molOri.GetAtoms())
                bond_num=len(molOri.GetBonds())
                aclas_num=len(atom_classes)
                bclas_num=len(bond_classes)
            if ac_b_smilesnotsame_writJson:
                b_datadir=  os.path.join(view_dirb2, f"{file_base}.json")
                ac_datadir=  os.path.join(view_dirac2, f"{file_base}.json")
                b_datadir_img=  os.path.join(view_dirb2, f"{file_base}.png")
                ac_datadir_img=  os.path.join(view_dirac2, f"{file_base}.png")
                shutil.copy(image_path, b_datadir_img)
                with open(ac_datadir,"w") as wf1:
                        json.dump(ac_dict, wf1, indent=2)
                with open(b_datadir,"w") as wf2:
                        json.dump(b_dict, wf2, indent=2)
    if getfpsim:
        if has_ref:
            if sameWithOutStero or sameWithOutStero_exp or logest_flag or logest_flagHfix:
                simRD+=1
                simRDlist.append(1)
                sim+=1
            else:
                try:
                    morgan_tani, tanimoto=molfpsim(SMILESori, expandStero_smi)
                    simRD+=tanimoto
                    simRDlist.append(tanimoto)
                    sim+=morgan_tani
                    print([tanimoto,morgan_tani],'smileNotsame!!!')
                except Exception as e:
                    print(f"mol to fingerprint erros")
                    simRD+=0
                    simRDlist.append(0)
                    sim+=0
        else:
            dfm+=1
    if view_dirac_tmp_debug:
        tmp_imgPath = os.path.join(view_dirac_tmp, f"{file_base}.png")
        if os.path.exists(image_path):
            shutil.move(image_path, tmp_imgPath)
if getacc:
    fm=len(smiles_data)
    eval_denom = max(fm, 1)
    if getfpsim:
        fpsim_denom = max(fm - dfm, 1)
        sim_100 = 100*sim/fpsim_denom
        simrd100 = 100*simRD/fpsim_denom
        tanimoto_to_one = mysum / fpsim_denom
        tanimoto_one_count = len(
        [tanimoto for tanimoto in simRDlist if float(tanimoto) == 1.0])
        tanimoto_to_one2 = tanimoto_one_count / fpsim_denom
        flogout.write(f"平均相似度RDKITfngerprint{simrd100}%, tanimoto@1.0为{tanimoto_to_one * 100}%\n")
        flogout.write(f"平均相似度Moganfngerprint{sim_100}%,  tanimoto2@1.0为{tanimoto_to_one2 * 100}%\n")
        flogout.write(f"avarage similarity morgan tanimoto: RDKFp tanimoto:: {sim_100}%,  {simrd100}%  \n")
    correct_pct = 100*mysum/eval_denom
    flogout.write(f"rdkit concanlized==smiles:{correct_pct}%\n")
    flogout.write(f"failed:{len(failed)}\n totoal saved in csv : {fm}\n")
    flogout.write(f'I2M@@:: match--{mysum},unmatch--{len(mydiff)},failed--{len(failed)},correct %{correct_pct} \n')
    if fm > 0:
        evaluator = SmilesEvaluator(smiles_data['SMILESori'], tanimoto=True)
        res_pre=evaluator.evaluate(smiles_data['SMILESpre'])
        res_exp=evaluator.evaluate(smiles_data['SMILESexp'])
        flogout.write(f'MolScribe style evaluation@SMILESpre:: {str(res_pre)} \n')
        flogout.write(f'MolScribe style evaluation@SMILESexp:: {str(res_exp)} \n')
    flogout.write(f'eligible samples evaluated: {len(rows_check)}\n')
    flogout.write(f'none molecule images skipped: {len(none_mol_images)}\n')
    flogout.close()
print(f"########################## final output ##########################")    
print(f"acc results saved into:\n{acc_summary}")
print(f"will save {fm} dataframe into csv") 
smiles_data.to_csv(outcsv_filename, index=False)            
if len(none_mol_images)>0:
    none_mol_csv = os.path.join(src_dir, f"{da}_none_mol_images.csv")
    none_mol_df = pd.DataFrame(
        none_mol_images,
        columns=[
            'id', 'file_name', 'file_base', 'file_path', 'SMILESori', 'reason',
            'atom1_idx', 'atom2_idx', 'atom2_1_idx',
            'atom_count', 'bond_count', 'charge_count', 'other_atom_count', 'element_atom_count'
        ]
    )
    if not none_mol_df.empty:
        none_mol_df = none_mol_df.drop_duplicates(subset=['file_name'], keep='first')
    none_mol_df.to_csv(none_mol_csv, index=False)
    print(f"none molecule images skipped: {len(none_mol_df)}")
    print(f"none molecule image list saved into:\n{none_mol_csv}")
print(f"miss json files number {len(miss_filejs)}")
print(f"failed image id base list {failed_fb}\n numbers {len(failed_fb)}")
failedlist=f"{args.dataname}_failedList.pkl"
try:
    import pickle
    with open(failedlist, 'wb') as f:
        pickle.dump(failed, f)
except (pickle.PicklingError, IOError) as e:
    print(f"Error saving list: {e}")
def release_ocr(ocr_instance):
    if hasattr(ocr_instance, 'detector'):
        ocr_instance.detector = None
    if hasattr(ocr_instance, 'recognizer'):
        ocr_instance.recognizer = None
    if hasattr(ocr_instance, 'cls'):
        ocr_instance.cls = None
release_ocr(ocr)
del ocr
release_ocr(ocr2)
del ocr2