import logging
import os
import json
import matplotlib.pyplot as plt
from collections import OrderedDict
import numpy as np

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, hooks, launch
from detectron2.evaluation import COCOEvaluator, inference_on_dataset, verify_results
from detectron2.modeling import GeneralizedRCNNWithTTA
from detectron2.data.datasets.coco import load_coco_json

# 宣告類別名稱
CLASS_NAMES = ["Car", "Van", "Truck", "Pedestrian", "Person_sitting", "Cyclist", "Tram", "Misc"]

# 設定資料集路徑
DATASET_ROOT = 'E:/detectron2/datasets/kitti'
TRAIN_PATH = os.path.join(DATASET_ROOT, 'images/train')
VAL_PATH = os.path.join(DATASET_ROOT, 'images/val')

TRAIN_JSON = os.path.join(DATASET_ROOT, 'json/train/train_corrected.json')
VAL_JSON = os.path.join(DATASET_ROOT, 'json/val/val_corrected.json')

# 註冊資料集
def plain_register_dataset():
    DatasetCatalog.register("coco_my_train", lambda: load_coco_json(TRAIN_JSON, TRAIN_PATH))
    MetadataCatalog.get("coco_my_train").set(thing_classes=CLASS_NAMES, evaluator_type='coco',
                                             json_file=TRAIN_JSON, image_root=TRAIN_PATH)

    DatasetCatalog.register("coco_my_val", lambda: load_coco_json(VAL_JSON, VAL_PATH))
    MetadataCatalog.get("coco_my_val").set(thing_classes=CLASS_NAMES, evaluator_type='coco',
                                           json_file=VAL_JSON, image_root=VAL_PATH)

class CustomCOCOEvaluator(COCOEvaluator):
    def __init__(self, dataset_name, cfg, distributed, output_dir=None):
        super().__init__(dataset_name, cfg, distributed, output_dir)
        self.predictions = []
        self._output_dir = output_dir

    def process(self, inputs, outputs):
        super().process(inputs, outputs)
        for output in outputs:
            instances = output["instances"]
            if instances.has("scores"):
                self.predictions.append(instances.get("scores").cpu().numpy())

    def evaluate(self):
        results = super().evaluate()
        self.plot_precision_recall()
        self.plot_probability_histogram()
        return results

    def plot_precision_recall(self):
        # Placeholder for actual precision-recall plotting
        precisions = np.linspace(0, 1, num=100)
        recalls = np.linspace(1, 0, num=100)
        plt.plot(recalls, precisions, marker='.')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.savefig(os.path.join(self._output_dir, "precision_recall_curve.png"))
        plt.close()

    def plot_probability_histogram(self):
        if len(self.predictions) == 0:
            return
        probs = np.concatenate(self.predictions)
        plt.hist(probs, bins=50, range=(0, 1), alpha=0.75, color='blue', edgecolor='black')
        plt.xlabel('Probability')
        plt.ylabel('Frequency')
        plt.title('Probability of Ground Truth Class')
        plt.savefig(os.path.join(self._output_dir, "probability_histogram.png"))
        plt.close()

class Trainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        return CustomCOCOEvaluator(dataset_name, cfg, True, output_folder)

    @classmethod
    def test_with_TTA(cls, cfg, model):
        logger = logging.getLogger("detectron2.trainer")
        logger.info("Running inference with test-time augmentation ...")
        model = GeneralizedRCNNWithTTA(cfg, model)
        evaluators = [
            cls.build_evaluator(cfg, name, output_folder=os.path.join(cfg.OUTPUT_DIR, "inference_TTA"))
            for name in cfg.DATASETS.TEST
        ]
        res = cls.test(cfg, model, evaluators)
        res = OrderedDict({k + "_TTA": v for k, v in res.items()})
        return res

def setup(args):
    plain_register_dataset()

    cfg = get_cfg()
    args.config_file = "E:/testdect/detectron2/configs/COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"
    #args.config_file = "E:/testdect/detectron2/configs/COCO-Detection/retinanet_R_50_FPN_3x.yaml"
    #args.config_file = "E:/testdect/detectron2/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
    
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)

    cfg.DATASETS.TRAIN = ("coco_my_train",)
    cfg.DATASETS.TEST = ("coco_my_val",)
    cfg.DATALOADER.NUM_WORKERS = 4

    cfg.INPUT.CROP.ENABLED = True
    cfg.INPUT.MAX_SIZE_TRAIN = 640
    cfg.INPUT.MAX_SIZE_TEST = 640
    cfg.INPUT.MIN_SIZE_TRAIN = (512, 768)
    cfg.INPUT.MIN_SIZE_TEST = 640
    cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING = 'range'

    cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(CLASS_NAMES)

    cfg.MODEL.WEIGHTS ="E:/testdect/detectron2/modelweight/model_final_280758.pkl" 
    # retinanet : cfg.MODEL.WEIGHTS ="E:/testdect/detectron2/modelweight/model_final_5bd44e.pkl"
    #maskrcnn : cfg.MODEL.WEIGHTS = "E:/testdect/detectron2/modelweight/model_final_f10217.pkl"
    
    cfg.SOLVER.IMS_PER_BATCH = 4

    ITERS_IN_ONE_EPOCH = int(9000 / cfg.SOLVER.IMS_PER_BATCH)
    cfg.SOLVER.MAX_ITER = (ITERS_IN_ONE_EPOCH * 12) - 1
    cfg.SOLVER.BASE_LR = 0.002
    cfg.SOLVER.MOMENTUM = 0.9
    cfg.SOLVER.WEIGHT_DECAY = 0.0001
    cfg.SOLVER.WEIGHT_DECAY_NORM = 0.0
    cfg.SOLVER.GAMMA = 0.1
    cfg.SOLVER.STEPS = (7000,)
    cfg.SOLVER.WARMUP_FACTOR = 1.0 / 1000
    cfg.SOLVER.WARMUP_ITERS = 1000
    cfg.SOLVER.WARMUP_METHOD = "linear"
    cfg.SOLVER.CHECKPOINT_PERIOD = ITERS_IN_ONE_EPOCH - 1
    cfg.TEST.EVAL_PERIOD = ITERS_IN_ONE_EPOCH

    cfg.MODEL.DEVICE = 'cuda'

    cfg.freeze()
    default_setup(cfg, args)
    return cfg

def main(args):
    cfg = setup(args)

    if args.eval_only:
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(cfg.MODEL.WEIGHTS, resume=args.resume)
        res = Trainer.test(cfg, model)
        if cfg.TEST.AUG.ENABLED:
            res.update(Trainer.test_with_TTA(cfg, model))
        if comm.is_main_process():
            verify_results(cfg, res)
        return res

    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    if cfg.TEST.AUG.ENABLED:
        trainer.register_hooks([hooks.EvalHook(0, lambda: trainer.test_with_TTA(cfg, trainer.model))])
    return trainer.train()

def invoke_main() -> None:
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    invoke_main()  # pragma: no cover
