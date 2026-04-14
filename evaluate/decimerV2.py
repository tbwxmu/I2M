import argparse
import os
import re
import sys
'''
Please install the DECIMER package or python environment with DECIMER.
This script only work after installing DECIMER !!!!
run as :  python decimerV2.py -gi 3 -da chemvlocr
'''

# Set runtime limits before importing DECIMER / TensorFlow.
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "2")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "2")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")


parser = argparse.ArgumentParser()
parser.add_argument("--dataname", "-da", type=str, default=None)
parser.add_argument("--gpuid", "-gi", type=str, default=None)
parser.add_argument("--number", "-n", type=int, default=None)
args, unknown = parser.parse_known_args()

if args.gpuid is not None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpuid)
else:
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")

from DECIMER import predict_SMILES
import pandas as pd
from rdkit import Chem
from rdkit import DataStructs
from rdkit import RDLogger
from rdkit.Chem import AllChem
from rdkit.Chem.MolStandardize import rdMolStandardize

RDLogger.DisableLog("rdApp.*")

I2M_SOLVER_DIR = "../src/solver"
if I2M_SOLVER_DIR not in sys.path:
    sys.path.append(I2M_SOLVER_DIR)

# from evaluate import SmilesEvaluator


VALID_ELEMENTS = {
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
    "Rg", "Cn", "Nh", "Fl", "Mc", "Lv", "Ts", "Og",
}

PSEUDO_ATOM_PATTERNS = [
    (r"\[R'+\]", "[*]"),
    (r"\[R\d*\]", "[*]"),
    (r"\[(Me|Et|Pr|Bu|Ph|Bn|Ts|Tf|Boc|Fmoc)\]", "[*]"),
    (r"\[(A|M|Q|X|Xa|Y|Z|L)\]", "[*]"),
]


def normalize_bracket_token(match):
    token = match.group(0)
    inner = token[1:-1]
    if "*" in inner:
        return token

    atom_match = re.match(r"^\d*([A-Z][a-z]?|[bcnops]|se|as)", inner)
    if atom_match is None:
        return "[*]"

    symbol = atom_match.group(1)
    normalized = symbol[0].upper() + symbol[1:].lower()
    aromatic_ok = symbol in {"b", "c", "n", "o", "p", "s", "se", "as"}
    if normalized in VALID_ELEMENTS or aromatic_ok:
        return token
    return "[*]"


def replace_problem_atoms_with_dummy(mol):
    problems = Chem.DetectChemistryProblems(mol)
    if not problems:
        return None, None

    rw_mol = Chem.RWMol(mol)
    replaced = []
    for problem in problems:
        if not hasattr(problem, "GetAtomIdx"):
            continue
        atom_idx = problem.GetAtomIdx()
        atom = rw_mol.GetAtomWithIdx(atom_idx)
        atom.SetAtomicNum(0)
        atom.SetFormalCharge(0)
        atom.SetNoImplicit(True)
        atom.SetNumExplicitHs(0)
        replaced.append((atom_idx, problem.GetType()))

    if not replaced:
        return None, None

    repaired = rw_mol.GetMol()
    try:
        Chem.SanitizeMol(repaired)
        return repaired, replaced
    except Exception:
        return None, None


def mol_from_smiles_with_fallback(smiles):
    if not isinstance(smiles, str) or not smiles.strip():
        return None, None, "empty_smiles"

    raw_smiles = smiles.strip()
    candidates = [raw_smiles]

    normalized = raw_smiles
    for pattern, replacement in PSEUDO_ATOM_PATTERNS:
        normalized = re.sub(pattern, replacement, normalized)
    normalized = re.sub(r"\[[^\[\]]+\]", normalize_bracket_token, normalized)
    if normalized != raw_smiles:
        candidates.append(normalized)

    seen = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)

        mol = Chem.MolFromSmiles(candidate)
        if mol is not None:
            reason = "direct" if candidate == raw_smiles else "token_replaced"
            return mol, candidate, reason

        try:
            unsanitized = Chem.MolFromSmiles(candidate, sanitize=False)
        except Exception:
            unsanitized = None
        if unsanitized is None:
            continue

        repaired, replaced = replace_problem_atoms_with_dummy(unsanitized)
        if repaired is not None:
            reason = "dummy_atom_repaired"
            if candidate != raw_smiles:
                reason = f"{reason}+token_replaced"
            return repaired, candidate, f"{reason}:{replaced}"

    return None, normalized, "rdkit_parse_failed"


def standardize_mol_smiles(mol, keep_isomeric=True):
    smiles = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=keep_isomeric)
    return rdMolStandardize.StandardizeSmiles(smiles)


def get_similarity_scores(mol1, mol2):
    morgan1 = AllChem.GetMorganFingerprint(mol1, 3, useChirality=True)
    morgan2 = AllChem.GetMorganFingerprint(mol2, 3, useChirality=True)
    dice = DataStructs.DiceSimilarity(morgan1, morgan2)
    fp1 = Chem.RDKFingerprint(mol1)
    fp2 = Chem.RDKFingerprint(mol2)
    rdk = DataStructs.FingerprintSimilarity(fp1, fp2)
    return dice, rdk


def molecules_match_for_fpsim(mol1, mol2):
    exact_smi1 = Chem.MolToSmiles(mol1, canonical=True, isomericSmiles=True)
    exact_smi2 = Chem.MolToSmiles(mol2, canonical=True, isomericSmiles=True)
    if exact_smi1 == exact_smi2:
        return True, exact_smi1, exact_smi2, None, None

    std_smi1 = standardize_mol_smiles(mol1, keep_isomeric=True)
    std_smi2 = standardize_mol_smiles(mol2, keep_isomeric=True)
    if std_smi1 == std_smi2:
        return True, exact_smi1, exact_smi2, std_smi1, std_smi2

    achiral_smi1 = standardize_mol_smiles(mol1, keep_isomeric=False)
    achiral_smi2 = standardize_mol_smiles(mol2, keep_isomeric=False)
    if achiral_smi1 == achiral_smi2:
        return True, exact_smi1, exact_smi2, achiral_smi1, achiral_smi2

    return False, exact_smi1, exact_smi2, std_smi1, std_smi2


def main():
    das = [
        "acs",
    ]
    if args.dataname:
        das = [args.dataname]

    for da in das:
        print(da)
        csv_path = f"/recovery/bo/pys/i2m_data/real/{da}.csv"#edit with your path
        acs_dir = f"/recovery/bo/pys/i2m_data/real/{da}"##edit with your path

        df = pd.read_csv(csv_path, header=0)
        if args.number is not None:
            df = df.head(args.number).copy()

        exact_match_count = 0
        standardized_match_count = 0
        achiral_match_count = 0
        morgan_similarity_sum = 0.0
        rdk_similarity_sum = 0.0
        rdk_similarity_list = []
        recovered_count = 0
        fpsim_invalid_ref_count = 0
        failed = []
        raw_diff = []
        standardized_diff = []
        achiral_diff = []
        eval_gold_smiles = []
        eval_pred_smiles = []
        eval_exp_smiles = []

        output_path = f"{da}V2out.txt"
        with open(output_path, "w", encoding="utf-8") as flogout:
            for i, row in df.iterrows():
                if i % 20 == 0:
                    print(f"[{i + 1}/{len(df)}] {da}")

                png_filename = str(df.loc[i, "file_path"]).split("/")[-1]
                smiles = df.loc[i, "SMILES"]
                img_path = os.path.join(acs_dir, png_filename)

                try:
                    smiles_pred = predict_SMILES(img_path)
                except Exception as e:
                    print(e)
                    smiles_pred = None

                eval_gold_smiles.append(smiles if isinstance(smiles, str) else "")
                eval_pred_smiles.append(smiles_pred if isinstance(smiles_pred, str) else "")

                if not isinstance(smiles, str) or not isinstance(smiles_pred, str):
                    if not isinstance(smiles, str):
                        fpsim_invalid_ref_count += 1
                    eval_exp_smiles.append("")
                    print(f"smiles problems\n{smiles}\n{smiles_pred}\n{img_path}")
                    failed.append([smiles, smiles_pred, img_path, "non_string_smiles"])
                    continue

                mol1, smiles_ref_fixed, reason1 = mol_from_smiles_with_fallback(smiles)
                mol2, smiles_pred_fixed, reason2 = mol_from_smiles_with_fallback(smiles_pred)
                eval_exp_smiles.append(smiles_pred_fixed if isinstance(smiles_pred_fixed, str) else "")

                recovered_ref = "dummy_atom_repaired" in reason1 or "token_replaced" in reason1
                recovered_pred = "dummy_atom_repaired" in reason2 or "token_replaced" in reason2
                if recovered_ref or recovered_pred:
                    recovered_count += 1

                if mol1 is None:
                    fpsim_invalid_ref_count += 1

                if (mol2 is None) or (mol1 is None):
                    print(
                        "get rdkit mol None\n"
                        f"ref: {smiles}\n"
                        f"pred: {smiles_pred}\n"
                        f"ref_fixed: {smiles_ref_fixed}\n"
                        f"pred_fixed: {smiles_pred_fixed}\n"
                        f"ref_reason: {reason1}\n"
                        f"pred_reason: {reason2}\n"
                        f"{img_path}"
                    )
                    failed.append([
                        smiles,
                        smiles_pred,
                        img_path,
                        f"ref={reason1};pred={reason2}",
                    ])
                    continue

                try:
                    is_fp_exact, rdk_smi1, rdk_smi2, std_smi1, std_smi2 = molecules_match_for_fpsim(
                        mol1, mol2
                    )
                except Exception as e:
                    print(f"molecule canonicalization errors: {e}")
                    failed.append([smiles, smiles_pred, img_path, "canonicalization_error"])
                    continue

                if is_fp_exact:
                    tanimoto = 1.0
                    morgan_tani = 1.0
                else:
                    try:
                        morgan_tani, tanimoto = get_similarity_scores(mol1, mol2)
                    except Exception as e:
                        print(f"mol to fingerprint errors: {e}")
                        tanimoto = 0.0
                        morgan_tani = 0.0

                rdk_similarity_sum += tanimoto
                morgan_similarity_sum += morgan_tani
                rdk_similarity_list.append(tanimoto)

                if rdk_smi1 == rdk_smi2:
                    exact_match_count += 1
                else:
                    raw_diff.append([smiles, smiles_pred, img_path, rdk_smi1, rdk_smi2])

                try:
                    if std_smi1 is None or std_smi2 is None:
                        std_smi1 = standardize_mol_smiles(mol1, keep_isomeric=True)
                        std_smi2 = standardize_mol_smiles(mol2, keep_isomeric=True)
                    achiral_smi1 = standardize_mol_smiles(mol1, keep_isomeric=False)
                    achiral_smi2 = standardize_mol_smiles(mol2, keep_isomeric=False)
                except Exception as e:
                    print(e)
                    failed.append([smiles, smiles_pred, img_path, "standardize_error"])
                    continue

                if std_smi1 == std_smi2:
                    standardized_match_count += 1
                else:
                    standardized_diff.append([smiles, smiles_pred, img_path, std_smi1, std_smi2])

                if achiral_smi1 == achiral_smi2:
                    achiral_match_count += 1
                else:
                    achiral_diff.append([smiles, smiles_pred, img_path, achiral_smi1, achiral_smi2])

            total = len(df)
            exact_acc = exact_match_count / total if total else 0.0
            # std_acc = standardized_match_count / total if total else 0.0
            # achiral_acc = achiral_match_count / total if total else 0.0
            # fpsim_denom = max(total - fpsim_invalid_ref_count, 1)
            # # sim_100 = 100 * morgan_similarity_sum / fpsim_denom
            # # simrd100 = 100 * rdk_similarity_sum / fpsim_denom
            # # tanimoto_to_one = exact_match_count / fpsim_denom
            # # tanimoto_one_count = len(
            # #     [tanimoto for tanimoto in rdk_similarity_list if float(tanimoto) == 1.0]
            # # )
            # tanimoto_to_one2 = tanimoto_one_count / fpsim_denom

            # flogout.write(f"rdkit canonical isomeric match:{100 * exact_acc}%\n")
            # flogout.write(f"rdMolStandardize.StandardizeSmiles:{std_acc}\n")
            # flogout.write(f"after StandardizeSmiles no stereo:{achiral_acc}\n")
            # flogout.write(
            #     f"平均相似度RDKITfngerprint{simrd100}%, tanimoto@1.0为{tanimoto_to_one * 100}%\n"
            # )
            # flogout.write(
            #     f"平均相似度Moganfngerprint{sim_100}%, tanimoto2@1.0为{tanimoto_to_one2 * 100}%\n"
            # )
            # flogout.write(
            #     f"sum,{exact_match_count},{standardized_match_count},{achiral_match_count}\n"
            # )
            # flogout.write(
            #     f"diffs nums:{len(raw_diff)},{len(standardized_diff)},{len(achiral_diff)}\n"
            # )
            # flogout.write(f"failed:{len(failed)}\n")
            # flogout.write(f"fpsim_invalid_ref:{fpsim_invalid_ref_count}\n")
            # flogout.write(f"recovered_with_dummy_or_token_replace:{recovered_count}\n")
            # flogout.write(
            #     f"ava similarity morgan dice / RDKFp tanimoto: {sim_100}%, {simrd100}%\n"
            # )
            flogout.write(
                f"DecimerV2@@{da}:: match--{exact_match_count},"
                f"unmatch--{len(raw_diff)},failed--{len(failed)},"
                f"correct %{100 * exact_acc}\n"
            )

            if failed:
                flogout.write("\nfailed examples:\n")
                for item in failed[:20]:
                    flogout.write(f"{item}\n")

            if standardized_diff:
                flogout.write("\nstandardized diff examples:\n")
                for item in standardized_diff[:20]:
                    flogout.write(f"{item}\n")

            # if total > 0:
            #     eval_workers = min(4, os.cpu_count() or 1)
            #     try:
            #         evaluator = SmilesEvaluator(eval_gold_smiles, num_workers=eval_workers, tanimoto=True)
            #         res_pre = evaluator.evaluate(eval_pred_smiles)
            #         res_exp = evaluator.evaluate(eval_exp_smiles)
            #         flogout.write(
            #             f"\nMolScribe style evaluation@SMILESpre:: {str(res_pre)} \n"
            #         )
            #         flogout.write(
            #             f"MolScribe style evaluation@SMILESexp:: {str(res_exp)} \n"
            #         )
            #     except Exception as e:
            #         flogout.write(f"\nMolScribe style evaluation failed: {e}\n")

        print(f"{da} dataset done")

    print("all done")


if __name__ == "__main__":
    main()
