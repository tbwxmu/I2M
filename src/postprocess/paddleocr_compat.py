import os
os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
from paddleocr import PaddleOCR
def get_paddleocr_version():
    try:
        import paddleocr
        return paddleocr.__version__
    except:
        return "unknown"
def create_ocr(**kwargs):
    version = get_paddleocr_version()
    is_new_version = version.startswith("3.")
    if is_new_version:
        rename_map = {
            'use_angle_cls': 'use_textline_orientation',
            'det_db_thresh': 'text_det_thresh',
            'det_db_box_thresh': 'text_det_box_thresh',
            'det_db_unclip_ratio': 'text_det_unclip_ratio',
            'det_limit_side_len': 'text_det_limit_side_len',
            'drop_score': 'text_rec_score_thresh',
            'det_model_dir': 'text_detection_model_dir',
            'rec_model_dir': 'text_recognition_model_dir',
            'cls_model_dir': 'textline_orientation_model_dir',
            'max_batch_size': 'text_recognition_batch_size',
        }
        deprecated_params = [
            'use_gpu',
            'show_log',
            'use_debug',
            'use_dilation',
            'det_db_score_mode',
            'det',
            'image_orientation',
            'rec_image_inverse',
            'rec_algorithm',
            'rec_image_shape',
            'max_text_length',
        ]
        filtered_kwargs = {}
        for key, value in kwargs.items():
            if key in deprecated_params:
                continue
            filtered_kwargs[rename_map.get(key, key)] = value
        if filtered_kwargs.get('det') is False:
            filtered_kwargs.pop('text_detection_model_dir', None)
            filtered_kwargs['use_textline_orientation'] = filtered_kwargs.get('use_textline_orientation', True)
        return PaddleOCR(**filtered_kwargs)
    return PaddleOCR(**kwargs)
def get_ocr_compatible(force_cpu=False, **kwargs):
    default_params = {
        'use_angle_cls': True,
        'lang': 'en',
        'det_db_thresh': 0.5,
        'det_db_box_thresh': 0.5,
        'det_db_unclip_ratio': 1.5,
        'use_dilation': False,
        'det_db_score_mode': 'fast',
        'drop_score': 0.5,
        'det_limit_side_len': 2500,
        'max_batch_size': 50,
        'show_log': False,
        'det_model_dir': os.path.dirname(__file__) + "/../../data/external/paddleocr/en_PP-OCRv3_det_infer/",
        'rec_model_dir': os.path.dirname(__file__) + "/../../data/external/paddleocr/en_PP-OCRv3_rec_infer/",
        'cls_model_dir': os.path.dirname(__file__) + "/../../data/external/paddleocr/ch_ppocr_mobile_v2.0_cls_infer",
        'use_gpu': not(force_cpu),
    }
    default_params.update(kwargs)
    return create_ocr(**default_params)
def get_ocr_recognition_only_compatible(force_cpu=False, **kwargs):
    default_params = {
        'lang': 'en',
        'drop_score': 1e-20,
        'image_orientation': True,
        'rec_image_inverse': False,
        'rec_algorithm': 'CRNN',
        'det': False,
        'rec_image_shape': [1, 50, 50],
        'use_angle_cls': True,
        'max_text_length': 60,
        'show_log': False,
        'det_model_dir': os.path.dirname(__file__) + "/../../data/external/paddleocr/en_PP-OCRv3_det_infer/",
        'rec_model_dir': os.path.dirname(__file__) + "/../../data/external/paddleocr/en_PP-OCRv3_rec_infer/",
        'cls_model_dir': os.path.dirname(__file__) + "/../../data/external/paddleocr/ch_ppocr_mobile_v2.0_cls_infer",
        'use_gpu': not(force_cpu),
    }
    default_params.update(kwargs)
    return create_ocr(**default_params)
def get_ocr_angle_compatible(force_cpu=False, **kwargs):
    default_params = {
        'use_angle_cls': True,
        'show_log': False,
        'rec_image_inverse': False,
        'det_model_dir': os.path.dirname(__file__) + "/../../data/external/paddleocr/en_PP-OCRv3_det_infer/",
        'rec_model_dir': os.path.dirname(__file__) + "/../../data/external/paddleocr/en_PP-OCRv3_rec_infer/",
        'cls_model_dir': os.path.dirname(__file__) + "/../../data/external/paddleocr/ch_ppocr_mobile_v2.0_cls_infer",
        'use_gpu': not(force_cpu),
    }
    default_params.update(kwargs)
    return create_ocr(**default_params)