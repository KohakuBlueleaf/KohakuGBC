# Detection selection for SAM 3, after a size preset (its box_score wins):
# gbc run configs/machine/<machine>.py configs/controls/mid.py configs/controls/sam3.py
# SAM 3 answers one box per instance: NMS only removes true duplicates (IoU 0.7; the PoC's 0.2
# removed overlapping objects, two people embracing at IoU 0.5), weak boxes go below 0.3 (on
# the 30 PoC images, below 0.3 a catch-all box around a whole region keeps appearing).

controls = {
    "box_score": 0.3,
    "nms_iou": 0.7,
    "nms_iou_single": 0.05,
    "drop_containers": True,
}
