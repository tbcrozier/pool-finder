1. Core Objective
Detect whether a property has a pool or no pool from overhead/satellite imagery.

2. Dataset Management
Maintain train, valid, and test datasets in a data/ folder:

train/ → for learning

valid/ → for tuning / early stopping

test/ → frozen set for unbiased final evaluation

Ensure no duplicates across splits.

Have a repeatable way to add new labeled images into the correct split without manual drag-and-drop.

3. Training Process
Use a pre-trained CNN (MobileNetV2) with transfer learning.

Apply data augmentation to make training more robust.

Save only the best model checkpoint (highest validation accuracy).

Log metrics (loss, accuracy) per epoch.

4. Evaluation
Evaluate the trained model on the frozen test set.

Output standard metrics:

Accuracy

Precision

Recall

F1 Score

Confusion Matrix

Track results over time so you can measure improvement.

5. Iterative Improvement Loop
Run predictions on unlabeled or misclassified images.

Visually review and correct labels.

Add corrected images back into the dataset.

Retrain and measure improvement on the test set.

6. Automation & Repeatability
Have one-command scripts (or Makefile) for:

Ingesting new labeled images (ingest_new.py)

Training (train_classifier.py)

Predicting (predict_images.py)

Evaluating (evaluate.py)

Keep a README.md with exact steps so you can resume after weeks away.

7. Optional Enhancements
Use Google Cloud Vision API once for bootstrapping labels if you want to speed up initial dataset building.

Consider a visual labeling tool (like FiftyOne) for faster review.

Store metrics & notes from each run in a runs.csv so you can track progress historically.

