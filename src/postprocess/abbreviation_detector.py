import numpy as np
import torch 
import cv2
from PIL import Image
from paddleocr import PaddleOCR
from .paddleocr_compat import get_ocr_recognition_only_compatible, get_ocr_compatible, get_ocr_angle_compatible
import multiprocessing
from tqdm import tqdm
import os
import json
from .utils_dataset import CaptionRemover, crop_tight
from .utils_image import resize_image, pad_image
def get_ocr_recognition_only(force_cpu = False):
    return get_ocr_recognition_only_compatible(force_cpu)
def get_ocr(force_cpu = False):
    return get_ocr_compatible(force_cpu)
def get_ocr_angle(force_cpu):
    return get_ocr_angle_compatible(force_cpu)
class AbbreviationDetector:
    def __init__(self, config, image_size = (1024, 1024), force_cpu = False, angle_recognition = False):
        self.image_size = image_size
        self.config = config
        self.border_size = 30
        self.force_cpu = force_cpu
        self.angle_recognition = angle_recognition
        torch.set_num_threads(1)
    def _detect_process(self, images_filenames_list, bonds_sizes_list):
        abbreviations_list = []
        for image_filename, bond_size in tqdm(zip(images_filenames_list, bonds_sizes_list), total=len(bonds_sizes_list)):
            pil_image = Image.open(image_filename)
            padded_image = pad_image(pil_image)
            padded_image = self.caption_remover(padded_image)
            pil_image = Image.fromarray(padded_image).convert('RGB')
            pil_image = crop_tight(pil_image)
            padded_image = pad_image(pil_image)
            image = np.array(padded_image, dtype=np.float32)/255
            image[image > 0.6] = 1. 
            image[image != 1.] = 0.
            image = np.stack((image, )*3, axis=-1)
            image = np.array(image*255, dtype=np.uint8)
            abbreviations_list.append(self(image, bond_size))
        return abbreviations_list
    def _detect_process_star(self, args):
        return self._detect_process(*args)
    def mp_run(self, images_filenames, bonds_sizes):
        if self.force_cpu:
            if len(images_filenames) < self.config["num_processes_mp"]:
                print("Abbreviation detector warning: Too much processes")
            images_filenames_split = np.array_split(images_filenames, self.config["num_processes_mp"])
            bonds_sizes_split = np.array_split(bonds_sizes, self.config["num_processes_mp"])
            args = [[images_filenames_split[process_index], bonds_sizes_split[process_index]] for process_index in range(self.config["num_processes_mp"])]
            pool = multiprocessing.Pool(self.config["num_processes_mp"])
            abbreviations_processes = pool.map(self._detect_process_star, args)
            pool.close()
            pool.join()
            abbreviations_list = []
            for index in range(self.config["num_processes_mp"]):
                abbreviations_list.extend(abbreviations_processes[index])
        else:
            abbreviations_list = self._detect_process(images_filenames, bonds_sizes)
        return abbreviations_list
    def __call__(self, image, bond_size = 100):
        self.image = image
        self.bond_size = bond_size
        self.filtered_image = self.filter_image(self.image, bond_size)
        self.set_letters_and_boxes()
        image_org, texts, scores, boxes = self.get_letter_predictions()
        abbreviations = [{
            "text": text,
            "box": [box[0], box[2]],
            "score": score,
        } for text, box, score in zip(texts, boxes, scores)]
        return abbreviations
    def set_letters_and_boxes(self, scope = 1.05, 
                              contour_method = cv2.RETR_EXTERNAL, 
                              contour_approx_mode = cv2.CHAIN_APPROX_SIMPLE):
        contours, _ = cv2.findContours(self.filtered_image.astype(np.uint8), contour_method, contour_approx_mode)
        boxes = []
        new_boxes = []
        letters = []
        for cts in contours:
            rect = cv2.minAreaRect(cts)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            boxes.append(box)
            xs = [i[0] for i in box]
            ys = [i[1] for i in box]
            x1 = min(xs)
            x2 = max(xs)
            y1 = min(ys)
            y2 = max(ys)
            new_boxes.append([[x1, y1],[x2, y2]])
        boxes = self.get_overlapping_boxes(new_boxes)
        new_boxes = []
        for box in boxes:
            xs = [i[0] for i in box]
            ys = [i[1] for i in box]
            x1 = min(xs)
            x2 = max(xs)
            y1 = min(ys)
            y2 = max(ys)
            center = (int((x1+x2)/2), int((y1+y2)/2))
            size = (int(scope*(x2-x1)), int(scope*(y2-y1)))
            cropped = cv2.getRectSubPix(self.image, size, center)/255
            cropped[cropped < 0.9] = 0
            cropped[cropped != 0] = 1
            letters.append(cropped)
            box2 = self.transform_box(box)
            new_boxes.append(box2)
        self.get_clean_mask(letters, new_boxes)
        self.letters = letters
        self.boxes = new_boxes
    def get_clean_mask(self, letters, boxes):
        mask = np.zeros(self.image_size, dtype=np.uint8)+1 
        for i in range(len(boxes)):
            minx = min(boxes[i][:, 1])
            miny = min(boxes[i][:, 0])
            mask[minx:minx+letters[i].shape[0], miny:miny+letters[i].shape[1]] = letters[i][:, :, 0]
        mask = mask*255
        self.image = np.stack((mask, )*3, axis=-1)
    def transform_box(self, box_to_transform, departing_box = None):
        box = [0, 1, 2, 3]
        if not isinstance(departing_box, np.ndarray):
            box[0] = [box_to_transform[0][0] + 1, box_to_transform[0][1] - 1]
            box[2] = [box_to_transform[1][0] + 1, box_to_transform[1][1] + 1]
            box[1] = [box_to_transform[1][0] + 1, box_to_transform[0][1] - 1]
            box[3] = [box_to_transform[0][0] - 1, box_to_transform[1][1] + 1]
        else:
            box[0] = [departing_box[0][0] + box_to_transform[0][0], departing_box[0][1] + box_to_transform[0][1]]
            box[2] = [departing_box[0][0] + box_to_transform[2][0], departing_box[0][1] + box_to_transform[2][1]]
            box[1] = [departing_box[0][0] + box_to_transform[1][0], departing_box[0][1] + box_to_transform[1][1]]
            box[3] = [departing_box[0][0] + box_to_transform[3][0], departing_box[0][1] + box_to_transform[3][1]]
        return np.array(box)
    def get_letter_predictions(self):
        texts, scores, new_boxes = [], [], []
        for i in range(len(self.boxes)):       
            if self.letters[i].shape[0] < self.bond_size/1.5:
                result = self.ocr_recognition_only.ocr(self.letters[i]*255, det = False, rec = True, cls = False)
                texts.append(result[0][0][0])
                scores.append(result[0][0][1])
                new_boxes.append(self.boxes[i])
            else:
                result = self.ocr.ocr(self.letters[i]*255, det = True, rec = True, cls = False)
                result = result[0]
                box = self.boxes[i]
                if result:
                    for line in result:
                        new_boxes.append(self.transform_box(line[0], box))
                        texts.append(line[1][0])
                        scores.append(line[1][1])
        return self.image, texts, scores, new_boxes, 
    def get_overlapping_boxes(self, boxes):
        def overlap(source, target):
            if (source[0][0] >= target[1][0] or target[0][0] >= source[1][0])                or (source[0][1] >= target[1][1] or target[0][1] >= source[1][1]):
                return False
            return True
        def get_overlaps(boxes, bounds, index):
            overlaps = []
            for a in range(len(boxes)):
                if a != index:
                    if overlap(bounds, boxes[a]):
                        overlaps.append(a)
            return overlaps
        merge_margin = 15
        finished = False
        highlight = [[0,0], [1,1]]
        points = [[[0,0]]]
        while not finished:
            finished = True
            index = len(boxes) - 1
            while index >= 0:
                curr = boxes[index]
                tl = curr[0][:]
                br = curr[1][:]
                tl[0] -= merge_margin
                tl[1] -= merge_margin
                br[0] += merge_margin
                br[1] += merge_margin
                overlaps = get_overlaps(boxes, [tl, br], index)
                if len(overlaps) > 0:
                    con = []
                    overlaps.append(index)
                    for ind in overlaps:
                        tl, br = boxes[ind]
                        con.append([tl])
                        con.append([br])
                    con = np.array(con)
                    x, y, w, h = cv2.boundingRect(con)
                    w -= 1
                    h -= 1
                    merged = [[x, y], [x + w, y + h]]
                    highlight = merged[:]
                    points = con
                    overlaps.sort(reverse = True)
                    for ind in overlaps:
                        del boxes[ind]
                    boxes.append(merged)
                    finished = False
                    break
                index -= 1
        return boxes
    def filter_image(self, image, bond_size = 100, pr_mode = 3, 
                kernel_size = 2, contour_method = cv2.RETR_EXTERNAL, 
                contour_approx_mode = cv2.CHAIN_APPROX_SIMPLE):
        img = image[:, :, 0]
        t1 = img.copy().astype(np.uint8)
        if pr_mode == 1:
            t1 = cv2.erode(t1, np.ones((kernel_size, kernel_size)))
        elif pr_mode == 2:
            t1 = cv2.morphologyEx(t1, cv2.MORPH_OPEN, np.ones((kernel_size, kernel_size)))
        else:
            t1 = cv2.morphologyEx(t1, cv2.MORPH_GRADIENT, np.ones((kernel_size, kernel_size)))
        img = cv2.cvtColor(np.zeros_like(t1), cv2.COLOR_GRAY2RGB)
        contours, _ = cv2.findContours(t1.astype(np.uint8), contour_method, contour_approx_mode)
        cnts = self.get_contours(contours, bond_size/1.5, bond_size*75, bond_size/2.5) 
        wc = cv2.drawContours(img, cnts, -1, (0, 255, 0), 2)
        wc =  cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        return cv2.bitwise_and(t1, t1, mask = wc)
    def get_contours(self, contours, beta1, beta2, alpha):
        cnts = []
        bsd = lambda x: np.abs(np.linalg.norm(x[0] - x[1]) - np.linalg.norm(x[0] - x[3]))
        for cnt in contours:
            _,radius = cv2.minEnclosingCircle(cnt)
            rect = cv2.minAreaRect(cnt)
            box = cv2.boxPoints(rect)
            if (beta1 < np.pi * radius**2 < beta2) and (bsd(box) < alpha):
                cnts.append(cnt)
        return cnts
class AbbreviationDetectorGPU(AbbreviationDetector):
    def __init__(self, config, image_size = (1024, 1024), force_cpu = False, angle_recognition = False):
        super(AbbreviationDetectorGPU, self).__init__(config, image_size, force_cpu, angle_recognition)
        self.caption_remover = CaptionRemover(force_cpu = False)
        self.ocr = get_ocr(force_cpu = False)
        self.ocr_recognition_only = get_ocr_recognition_only(force_cpu = False)
        if angle_recognition:
            self.ocr_angle = get_ocr_angle(force_cpu = False)
class AbbreviationDetectorCPU(AbbreviationDetector):
    def __init__(self, config, image_size = (1024, 1024), force_cpu = True, angle_recognition = False):
        super(AbbreviationDetectorCPU, self).__init__(config, image_size, force_cpu, angle_recognition)
class SpellingCorrector:
    def __init__(self, abbreviations_smiles_mapping=None):
        self.abbreviations = abbreviations_smiles_mapping
        with open(os.path.dirname(__file__) + "/../../data/ocr_mapping/ocr_atoms_classes_mapping.json") as file:
            ocr_atoms_classes_mapping = json.load(file)
        self.abbreviations = {**self.abbreviations, **ocr_atoms_classes_mapping} 
        self.set_costs()
    def __call__(self, predicted_abbreviation):       
        if predicted_abbreviation in self.abbreviations:
            return predicted_abbreviation
        score_min = 150
        indices_to_check = []
        for i, correct_abb in enumerate(self.abbreviations):
            levenshtein_score = lev(
                predicted_abbreviation, 
                correct_abb, 
                substitute_costs = self.substitute_costs, 
                insert_costs = self.insert_costs, 
                delete_costs = self.delete_costs
            )
            if levenshtein_score < score_min:
                score_min = levenshtein_score
                indices_to_check = [i]
            elif levenshtein_score == score_min:
                indices_to_check.append(i)
        if len(indices_to_check) == 1:
            return list(self.abbreviations.keys())[indices_to_check[0]]
        else:
            keys_candidates = [list(self.abbreviations.keys())[index] for index in indices_to_check]
            abbreviations_populations = {k: v["population"] for k, v in self.abbreviations.items() if k in keys_candidates}
            max_population = 0
            for abbreviation in abbreviations_populations:
                if abbreviations_populations[abbreviation] > max_population:
                    corrected_abbreviation = abbreviation
                    max_population = abbreviations_populations[abbreviation]
            print(f"Spelling correction problem: multiple abbreviation candidates for {predicted_abbreviation} have the same score: {keys_candidates}. {corrected_abbreviation} is chosen")
            return corrected_abbreviation
    def set_costs(self):
        self.insert_costs = np.ones(128, dtype=np.float64) 
        self.insert_costs[ord('2')] = 0.6
        self.insert_costs[ord('H')] = 0.8
        self.insert_costs[ord('C')] = 0.8
        self.insert_costs[ord('3')] = 0.8
        self.insert_costs[ord('(')] = 0.8
        self.insert_costs[ord(')')] = 0.8
        self.delete_costs = np.ones(128, dtype=np.float64)
        self.delete_costs[ord(' ')] = 0.5
        self.substitute_costs = np.ones((128, 128), dtype=np.float64)
        self.substitute_costs[ord('0'), ord('O')] = 0.1
        self.substitute_costs[ord('l'), ord('1')] = 0.3
        self.substitute_costs[ord(' '), ord('H')] = 0.5
        self.substitute_costs[ord('z'), ord('2')] = 0.5
        self.substitute_costs[ord('z'), ord('5')] = 0.6
        self.substitute_costs[ord('D'), ord('O')] = 0.5
        self.substitute_costs[ord('Q'), ord('O')] = 0.5
        self.substitute_costs[ord(','), ord('2')] = 0.5
        self.substitute_costs[ord('.'), ord('c')] = 0.5
        self.substitute_costs[ord(' '), ord('2')] = 0.6
        self.substitute_costs[ord('D'), ord('C')] = 0.5
        self.substitute_costs[ord('O'), ord('C')] = 0.7
        self.substitute_costs[ord('K'), ord('H')] = 0.5
        self.substitute_costs[ord('o'), ord('c')] = 0.5
        self.substitute_costs[ord('g'), ord('8')] = 0.5
        self.substitute_costs[ord('g'), ord('3')] = 0.6
        self.substitute_costs[ord('-'), ord('2')] = 0.5
        self.substitute_costs[ord('c'), ord('C')] = 0.4
        self.substitute_costs[ord('o'), ord('O')] = 0.4
        self.substitute_costs[ord('n'), ord('N')] = 0.4
        self.substitute_costs[ord('h'), ord('H')] = 0.4
        self.substitute_costs[ord('s'), ord('S')] = 0.4
        self.substitute_costs[ord('d'), ord('c')] = 0.5
        self.substitute_costs[ord('{'), ord('(')] = 0.5
        self.substitute_costs[ord('}'), ord(')')] = 0.5
        self.substitute_costs[ord('C'), ord('c')] = 0.8
        self.substitute_costs[ord('3'), ord('5')] = 0.8
        self.substitute_costs[ord('i'), ord('l')] = 0.5
        self.substitute_costs[ord('f'), ord('l')] = 0.5
        self.substitute_costs[ord('M'), ord('N')] = 0.5
        self.substitute_costs[ord('H'), ord('N')] = 0.8
        self.substitute_costs[ord('l'), ord('M')] = 0.5
        self.substitute_costs[ord('U'), ord('u')] = 0.5
        self.substitute_costs[ord('l'), ord('t')] = 0.5
        self.substitute_costs[ord('8'), ord('e')] = 0.5
        self.substitute_costs[ord('B'), ord('e')] = 0.5
        self.substitute_costs[ord('R'), ord('e')] = 0.5
        self.substitute_costs[ord('5'), ord('S')] = 0.5
        self.substitute_costs[ord('8'), ord('S')] = 0.5
        self.substitute_costs[ord('Z'), ord('z')] = 0.5
        self.substitute_costs[ord('l'), ord('i')] = 0.5
        self.insert_costs[ord('n')] = 0.8