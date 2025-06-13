import numpy as np
import pandas as pd
import cv2, os

''' =================================================================================
    Input  : manual labels of test-set
             output masks of model inference
    Output : comparison results between masks and labels 
    Metrics: Accuracy, Precision, Recall, F1-score, IoU        
    This code is for performance assessment via given masks and labels
================================================================================= '''

# Size of labels and masks
size = 512
# The path of input folder, including 'masks' and 'labels'
base_dir = "./Assessment/Assessment_Input"

# ========================= Test 70 ========================= #
labels_dir = os.path.join(base_dir, "Test70_labels/")
images_dir = os.path.join(base_dir, "Test70_masks/coralscop_prom")
output_dir = "./Assessment/Assessment_Output/Test70/coralscop_prom"

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Obtain names of labels and masks
image_files = sorted([f for f in os.listdir(images_dir) if f.endswith('.png')])
label_files = sorted([f for f in os.listdir(labels_dir) if f.endswith('.png')])

# Ensure the number of labels and masks is same
assert len(image_files) == len(label_files), "The number of labels and masks is different!!!"

visual = np.empty((size, size, 3)) # Visualization of comparsion results
result = []

# Iterate over all the mask and label files
for idx, (image_file, label_file) in enumerate(zip(image_files, label_files)):
    print(f"Processing file number: {idx + 1}")

    im_a_fname = os.path.join(images_dir, image_file)
    im_b_fname = os.path.join(labels_dir, label_file)

    im_a = cv2.imread(im_a_fname)
    im_b = cv2.imread(im_b_fname)

    TP = TN = FP = FN = 0

    # calculate metrics
    for i in range(size):
        for j in range(size):
            a = 1 if (int(im_a[i, j, 0]) + int(im_a[i, j, 1]) + int(im_a[i, j, 2]) >= 255 * 3 / 2) else 0
            b = 1 if (int(im_b[i, j, 0]) + int(im_b[i, j, 1]) + int(im_b[i, j, 2]) >= 255 * 3 / 2) else 0

            if a == 1 and b == 1:
                visual[i][j] = [255, 255, 255]
                TP += 1
            elif a == 1 and b == 0:
                visual[i][j] = [0, 0, 255]
                FP += 1
            elif a == 0 and b == 1:
                visual[i][j] = [255, 0, 0]
                FN += 1
            else:
                visual[i][j] = [0, 0, 0]
                TN += 1

    accuracy = (TP + TN) / float(TP + FP + TN + FN)
    if (TP + FP) == 0:
        precision = 1.0
    else:
        precision = TP / float(TP + FP)

    # Recall
    if (TP + FN) == 0:
        recall = 1.0
    else:
        recall = TP / float(TP + FN)

    # F1
    if (precision + recall) == 0:
        f_score = 0.0
    else:
        f_score = 2 * precision * recall / (precision + recall)

    # IoU
    if (TP + FP + FN) == 0:
        iou = 1.0
    else:
        iou = TP / float(TP + FP + FN)

    # precision = TP / float(TP + FN) if (TP + FN) > 0 else 0
    # recall = TP / float(TP + FP) if (TP + FP) > 0 else 0
    # f_score = (2 * recall * precision) / float(recall + precision) if (recall + precision) > 0 else 0
    # iou = TP / float(TP + FP + FN) if (TP + FP + FN) > 0 else 0

    output_visual_path = os.path.join(output_dir, image_file)
    cv2.imwrite(output_visual_path, visual)
    result.append([idx + 1, TP, FP, FN, TN, accuracy, precision, recall, f_score, iou])
    print('Accuracy', accuracy, 'precision', precision, 'recall', recall, 'f_score', f_score, 'IoU', iou)

# Calculate mean value of metrics
ave_result = np.mean(np.array(result), axis=0)
print('\nAVERAGE Accuracy', ave_result[5], 'precision', ave_result[6], 'recall', ave_result[7],
      'f_score', ave_result[8], 'mIoU', ave_result[9])

# Save results in .CSV file
df = pd.DataFrame(result, columns=["file_number", "TP", "FP", "FN", "TN", "Accuracy", "Precision", "Recall", "F-Score", "IoU"])
df.to_csv(os.path.join(output_dir, "result.csv"), index=False)

