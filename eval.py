import os

def parse_gt_line(line):
    parts = line.split(", ")
    x1, y1, x2, y2 = map(int, parts[:4])
    content = parts[4]
    return (x1, y1, x2, y2), content

def parse_result_line(line):
    x1, y1, x2, y2 = map(int, line.split(", "))
    return (x1, y1, x2, y2)

def calculate_iou(box1, box2):
    x1_max = max(box1[0], box2[0])
    y1_max = max(box1[1], box2[1])
    x2_min = min(box1[2], box2[2])
    y2_min = min(box1[3], box2[3])

    intersection = max(0, x2_min - x1_max) * max(0, y2_min - y1_max)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    union = box1_area + box2_area - intersection
    
    if union == 0:
        return 0
    return intersection / union

def evaluate(gt_folder, result_folder, iou_threshold=0.5):
    gt_files = sorted([f for f in os.listdir(gt_folder) if f.endswith('.txt')])
    result_files = sorted([f for f in os.listdir(result_folder) if f.endswith('.txt')])

    total_tp, total_fp, total_fn = 0, 0, 0

    for gt_file, result_file in zip(gt_files, result_files):
        with open(os.path.join(gt_folder, gt_file), 'r') as f_gt:
            gt_boxes = [parse_gt_line(line.strip())[0] for line in f_gt]

        with open(os.path.join(result_folder, result_file), 'r') as f_res:
            result_boxes = [parse_result_line(line.strip()) for line in f_res]

        tp, fp, fn = 0, 0, 0
        matched_gt = set()

        for res_box in result_boxes:
            match_found = False
            for i, gt_box in enumerate(gt_boxes):
                if i in matched_gt:
                    continue

                if calculate_iou(res_box, gt_box) >= iou_threshold:
                    tp += 1
                    match_found = True
                    matched_gt.add(i)
                    break

            if not match_found:
                fp += 1

        fn = len(gt_boxes) - tp

        total_tp += tp
        total_fp += fp
        total_fn += fn

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1_score:.4f}")

# Example usage:
print("assamese")
evaluate('assamese\est_gt', 'assamese\est_result')
