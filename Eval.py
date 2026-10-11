## Eval

# ============================================================
# ORDINARY K-MEANS++ ON UNLABELED DATA ONLY
#
# Clustering:
#     ONLY the 7000 unlabeled samples
#
# Evaluation:
#     ONLY the same 7000 unlabeled samples
#
# No labeled samples are used for clustering.
# ============================================================

import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.metrics import (
    accuracy_score,
    adjusted_rand_score,
    normalized_mutual_info_score,
    silhouette_score
)
from scipy.optimize import linear_sum_assignment


print("=" * 80)
print("ORDINARY K-MEANS++ — UNLABELED DATA ONLY")
print("=" * 80)


# ------------------------------------------------------------
# 1. Extract the 7000 unlabeled samples
# ------------------------------------------------------------

X_unknown = X_learned[unknown_indices]
y_unknown = y_split[unknown_indices]


print("\nUnlabeled data:")
print("X_unknown shape:", X_unknown.shape)
print("y_unknown shape:", y_unknown.shape)


# ------------------------------------------------------------
# 2. Run ordinary K-Means++
#
# IMPORTANT:
# K-Means sees ONLY the 7000 unlabeled samples.
# ------------------------------------------------------------

print("\nRunning ordinary K-Means++...")

kmeans = KMeans(
    n_clusters=14,
    init="k-means++",
    n_init=10,
    max_iter=300,
    random_state=42,
    algorithm="lloyd"
)

cluster_labels = kmeans.fit_predict(X_unknown)


print("✓ K-Means++ completed")


# ------------------------------------------------------------
# 3. Hungarian matching for clustering accuracy
#
# Cluster IDs are arbitrary, so we optimally map cluster IDs
# to true class IDs using the Hungarian algorithm.
# ------------------------------------------------------------

def clustering_accuracy(y_true, y_pred, n_classes=14):

    contingency = np.zeros(
        (n_classes, n_classes),
        dtype=np.int64
    )

    for true_label, pred_label in zip(
        y_true,
        y_pred
    ):
        contingency[
            pred_label,
            true_label
        ] += 1

    row_ind, col_ind = linear_sum_assignment(
        -contingency
    )

    matched_correct = contingency[
        row_ind,
        col_ind
    ].sum()

    accuracy = (
        matched_correct /
        len(y_true)
    )

    return accuracy, row_ind, col_ind, contingency


accuracy, row_ind, col_ind, contingency = (
    clustering_accuracy(
        y_unknown,
        cluster_labels,
        n_classes=14
    )
)


# ------------------------------------------------------------
# 4. Other clustering metrics
# ------------------------------------------------------------

ari = adjusted_rand_score(
    y_unknown,
    cluster_labels
)

nmi = normalized_mutual_info_score(
    y_unknown,
    cluster_labels
)

silhouette = silhouette_score(
    X_unknown,
    cluster_labels
)


# ------------------------------------------------------------
# 5. Print results
# ------------------------------------------------------------

print("\n" + "=" * 80)
print("ORDINARY K-MEANS++ RESULTS")
print("=" * 80)

print(f"Samples clustered : {len(X_unknown)}")
print(f"Number of clusters: 14")
print(f"Feature dimension  : {X_unknown.shape[1]}")

print(f"\nClustering Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")
print(f"ARI                : {ari:.4f}")
print(f"NMI                : {nmi:.4f}")
print(f"Silhouette         : {silhouette:.4f}")


# ------------------------------------------------------------
# 6. Class-wise accuracy after Hungarian matching
# ------------------------------------------------------------

cluster_to_class = {
    cluster: true_class
    for cluster, true_class
    in zip(row_ind, col_ind)
}

mapped_predictions = np.array([
    cluster_to_class.get(
        cluster,
        -1
    )
    for cluster in cluster_labels
])


print("\n" + "-" * 80)
print("CLASS-WISE ACCURACY — UNLABELED DATA ONLY")
print("-" * 80)

for class_id, class_name in enumerate(CLASS_ORDER):

    mask = (
        y_unknown == class_id
    )

    total = mask.sum()

    if total == 0:
        continue

    correct = np.sum(
        mapped_predictions[mask]
        == class_id
    )

    class_acc = correct / total

    print(
        f"{class_name:20s} "
        f"{correct:4d}/{total:4d} "
        f"({class_acc * 100:6.2f}%)"
    )

# ============================================================
# SEMI-SUPERVISED K-MEANS++ FOR GCD
#
# INITIALIZATION
# ----------------
# Known classes:
#     10 centers from the 7000 labeled samples
#     One mean center per known class
#
# Novel classes:
#     4 centers initialized using K-Means++ on the
#     7000 unlabeled samples
#
# CLUSTERING
# ----------------
# Run K-Means iterations using ALL 14000 samples.
#
# The 7000 labeled known samples are constrained to their
# known class during assignment.
#
# EVALUATION
# ----------------
# Evaluate ONLY the 7000 unlabeled samples.
# ============================================================

import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    silhouette_score
)

from scipy.optimize import linear_sum_assignment


print("=" * 80)
print("SEMI-SUPERVISED K-MEANS++ — GCD")
print("=" * 80)


# ============================================================
# 1. Prepare data
# ============================================================

X_all = X_learned
y_all = y_split

X_labeled = X_all[known_indices]
y_labeled = y_all[known_indices]

X_unlabeled = X_all[unknown_indices]
y_unlabeled = y_all[unknown_indices]


assert X_all.shape[0] == 14000


print("\nData:")
print("All data       :", X_all.shape)
print("Labeled data   :", X_labeled.shape)
print("Unlabeled data :", X_unlabeled.shape)


# ============================================================
# 2. Initialize centers for the 10 known classes
#
# Each known-class center is the mean of its 700 labeled
# samples.
# ============================================================

known_centers = []

print("\nKnown-class centers:")

for class_id, class_name in enumerate(KNOWN_CLASSES):

    class_mask = (
        y_labeled == class_id
    )

    class_features = (
        X_labeled[class_mask]
    )

    assert class_features.shape[0] == 700

    center = class_features.mean(
        axis=0
    )

    known_centers.append(center)

    print(
        f"{class_name:20s}: "
        f"{class_features.shape[0]} samples"
    )


known_centers = np.vstack(
    known_centers
)



# ============================================================
# 3. Initialize the 4 novel centers using K-Means++
#
# IMPORTANT:
# K-Means++ initialization is performed ONLY on the
# 7000 unlabeled samples.
# ============================================================

print("\nInitializing 4 novel centers using K-Means++...")

novel_kmeans = KMeans(
    n_clusters=5,
    init="k-means++",
    n_init=10,
    max_iter=1,
    random_state=42,
    algorithm="lloyd"
)

novel_kmeans.fit(
    X_unlabeled
)

novel_centers = novel_kmeans.cluster_centers_



# ============================================================
# 4. Combine the initial centers
#
# Center IDs:
#
# 0  - Accordion
# 1  - Acoustic_Guitar
# 2  - Banjo
# ...
# 9  - Horn
#
# 10 - first novel center
# 11 - second novel center
# 12 - third novel center
# 13 - fourth novel center
# ============================================================

centers = np.vstack([
    known_centers,
    novel_centers
])


assert centers.shape == (
    14,
    X_all.shape[1]
)


print("\n✓ Initial 14 centers created")


# ============================================================
# 5. Semi-supervised K-Means iterations
#
# All 14000 samples participate in center updates.
#
# However:
#     labeled known samples are forced to their true
#     known-class cluster.
# ============================================================

MAX_ITER = 300
TOL = 1e-6


print("\nRunning semi-supervised K-Means...")


for iteration in range(MAX_ITER):

    old_centers = centers.copy()


    # --------------------------------------------------------
    # Calculate squared Euclidean distances
    # --------------------------------------------------------

    distances = np.sum(
        (
            X_all[:, None, :]
            -
            centers[None, :, :]
        ) ** 2,
        axis=2
    )


    # --------------------------------------------------------
    # Normal nearest-center assignment for all samples
    # --------------------------------------------------------

    assignments = np.argmin(
        distances,
        axis=1
    )


    # --------------------------------------------------------
    # IMPORTANT:
    # Force the 7000 labeled samples to their known classes.
    # --------------------------------------------------------

    assignments[
        known_indices
    ] = y_all[
        known_indices
    ]


    # --------------------------------------------------------
    # Recompute all 14 centers using ALL 14000 samples
    # --------------------------------------------------------

    new_centers = np.zeros_like(
        centers
    )


    for cluster_id in range(14):

        cluster_mask = (
            assignments == cluster_id
        )

        if np.any(cluster_mask):

            new_centers[
                cluster_id
            ] = X_all[
                cluster_mask
            ].mean(axis=0)

        else:

            # Keep previous center if no sample is assigned.
            new_centers[
                cluster_id
            ] = centers[
                cluster_id
            ]


    centers = new_centers


    # --------------------------------------------------------
    # Check convergence
    # --------------------------------------------------------

    center_shift = np.max(
        np.linalg.norm(
            centers - old_centers,
            axis=1
        )
    )


    if (
        iteration == 0
        or (iteration + 1) % 10 == 0
    ):

        print(
            f"Iteration {iteration + 1:3d} | "
            f"Max center shift = {center_shift:.8f}"
        )


    if center_shift < TOL:

        print(
            f"\n✓ Converged at iteration "
            f"{iteration + 1}"
        )

        break

else:

    print(
        f"\nReached MAX_ITER={MAX_ITER}"
    )


# ============================================================
# 6. Final assignments
#
# Recalculate assignments using final centers.
#
# Labeled known samples remain constrained.
# ============================================================

final_distances = np.sum(
    (
        X_all[:, None, :]
        -
        centers[None, :, :]
    ) ** 2,
    axis=2
)


final_assignments = np.argmin(
    final_distances,
    axis=1
)


# Force labeled samples to their known classes
final_assignments[
    known_indices
] = y_all[
    known_indices
]


# ============================================================
# 7. Extract ONLY the 7000 unlabeled predictions
# ============================================================

pred_unlabeled = (
    final_assignments[
        unknown_indices
    ]
)


true_unlabeled = (
    y_all[
        unknown_indices
    ]
)



# ============================================================
# 8. Hungarian matching
#
# Cluster IDs are arbitrary.
#
# We optimally map the 14 cluster IDs to the 14 true
# class IDs using ONLY the 7000 unlabeled samples.
# ============================================================

contingency = np.zeros(
    (14, 14),
    dtype=np.int64
)


for predicted_cluster, true_class in zip(
    pred_unlabeled,
    true_unlabeled
):

    contingency[
        predicted_cluster,
        true_class
    ] += 1


row_ind, col_ind = (
    linear_sum_assignment(
        -contingency
    )
)


matched_correct = contingency[
    row_ind,
    col_ind
].sum()


unlabeled_accuracy = (
    matched_correct /
    len(true_unlabeled)
)



unlabeled_ari = (
    adjusted_rand_score(
        true_unlabeled,
        pred_unlabeled
    )
)


unlabeled_nmi = (
    normalized_mutual_info_score(
        true_unlabeled,
        pred_unlabeled
    )
)


# ============================================================
# 10. Silhouette
#
# Again: evaluation is ONLY on the 7000 unlabeled samples.
# ============================================================

unlabeled_silhouette = (
    silhouette_score(
        X_unlabeled,
        pred_unlabeled
    )
)


# ============================================================
# 11. Print final results
# ============================================================

print("\n" + "=" * 80)
print("SEMI-SUPERVISED K-MEANS++ RESULTS")
print("=" * 80)

print(
    f"Samples used for clustering : {len(X_all)}"
)

print(
    f"Labeled samples              : {len(X_labeled)}"
)

print(
    f"Unlabeled samples            : {len(X_unlabeled)}"
)

print(
    f"Number of clusters           : 14"
)

print(
    f"Feature dimension            : {X_all.shape[1]}"
)

print("\nEVALUATION — UNLABELED 7000 ONLY")

print(
    f"Clustering Accuracy : "
    f"{unlabeled_accuracy:.4f} "
    f"({unlabeled_accuracy * 100:.2f}%)"
)

print(
    f"ARI                 : "
    f"{unlabeled_ari:.4f}"
)

print(
    f"NMI                 : "
    f"{unlabeled_nmi:.4f}"
)

print(
    f"Silhouette          : "
    f"{unlabeled_silhouette:.4f}"
)


# ============================================================
# 12. Class-wise accuracy
# ============================================================

cluster_to_class = {
    cluster_id: class_id
    for cluster_id, class_id
    in zip(row_ind, col_ind)
}


mapped_predictions = np.array([
    cluster_to_class.get(
        cluster_id,
        -1
    )
    for cluster_id in pred_unlabeled
])


print("\n" + "-" * 80)
print("CLASS-WISE ACCURACY — UNLABELED DATA ONLY")
print("-" * 80)


for class_id, class_name in enumerate(
    CLASS_ORDER
):

    mask = (
        true_unlabeled == class_id
    )

    total = mask.sum()

    if total == 0:
        continue

    correct = np.sum(
        mapped_predictions[mask]
        == class_id
    )

    class_accuracy = (
        correct / total
    )

    print(
        f"{class_name:20s} "
        f"{correct:4d}/{total:4d} "
        f"({class_accuracy * 100:6.2f}%)"
    )

# ============================================================
# SEMI-SUPERVISED K-MEANS++ FOR GCD
#
# CORRECT INITIALIZATION:
#
# 1. Create 10 known-class centers from the 7000 labeled
#    samples using class-wise means.
#
# 2. Using ALL 14000 samples, perform K-Means++-style
#    initialization for the remaining 4 centers.
#
#    IMPORTANT:
#    The K-Means++ distances are calculated relative to
#    the already-created 10 known centers.
#
#    Therefore this is NOT equivalent to:
#
#        KMeans(n_clusters=4).fit(X_unlabeled)
#
# 3. After 14 initial centers exist:
#       10 known centers
#       4 novel centers
#
#    run semi-supervised K-Means iterations using ALL 14000
#    samples.
#
# 4. The 7000 labeled samples are constrained to their known
#    classes during every assignment step.
#
# 5. Evaluate ONLY the 7000 unlabeled samples.
#
# 6. Hungarian matching is used for clustering accuracy.
#
# 7. Plot 14 x 14 confusion matrix after Hungarian mapping.
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    silhouette_score,
    confusion_matrix
)

from scipy.optimize import linear_sum_assignment


print("=" * 90)
print("SEMI-SUPERVISED K-MEANS++ — CORRECT GCD INITIALIZATION")
print("=" * 90)


# ============================================================
# 1. Prepare all data
# ============================================================

X_all = np.asarray(X_learned, dtype=np.float64)
y_all = np.asarray(y_split, dtype=np.int64)


assert X_all.shape[0] == 14000
assert y_all.shape[0] == 14000


X_labeled = X_all[known_indices]
y_labeled = y_all[known_indices]

X_unlabeled = X_all[unknown_indices]
y_unlabeled = y_all[unknown_indices]


print("\nData:")
print(f"All samples       : {X_all.shape}")
print(f"Labeled samples   : {X_labeled.shape}")
print(f"Unlabeled samples : {X_unlabeled.shape}")


# ============================================================
# 2. Create the 10 known-class centers
#
# Each center is the mean of the 700 labeled samples belonging
# to that known class.
# ============================================================

known_centers = []

print("\n" + "-" * 90)
print("STEP 1 — KNOWN CLASS CENTERS")
print("-" * 90)


for class_id, class_name in enumerate(KNOWN_CLASSES):

    class_mask = (
        y_labeled == class_id
    )

    class_samples = X_labeled[
        class_mask
    ]

    assert class_samples.shape[0] == 700

    center = class_samples.mean(
        axis=0
    )

    known_centers.append(center)

    print(
        f"{class_id:2d} | "
        f"{class_name:20s} | "
        f"{class_samples.shape[0]:4d} samples"
    )


known_centers = np.vstack(
    known_centers
)


print(
    f"\nKnown centers shape: "
    f"{known_centers.shape}"
)


# ============================================================
# 3. K-Means++ initialization for the remaining 4 centers
#
# IMPORTANT:
#
# The candidate points are ALL 14000 samples.
#
# The initial set of centers already contains the 10 known
# class centers.
#
# At every step:
#
#     D(x) = distance from x to its nearest existing center
#
#     P(x) proportional to D(x)^2
#
# This is the K-Means++ D^2 sampling rule.
# ============================================================

print("\n" + "-" * 90)
print("STEP 2 — K-MEANS++ INITIALIZATION OF 4 REMAINING CENTERS")
print("-" * 90)

rng = np.random.default_rng(42)

# Start with the 10 known centers
initial_centers = known_centers.copy()


# ------------------------------------------------------------
# Calculate squared distance from every one of the 14000
# samples to the nearest known center.
# ------------------------------------------------------------

def squared_distance_to_nearest_center(
    X,
    centers
):

    distances = np.sum(
        (
            X[:, None, :]
            -
            centers[None, :, :]
        ) ** 2,
        axis=2
    )

    return np.min(
        distances,
        axis=1
    )


min_squared_distances = (
    squared_distance_to_nearest_center(
        X_all,
        initial_centers
    )
)


# Keep track of selected dataset rows so that the same
# sample is not selected twice.
selected_indices = set()


# ------------------------------------------------------------
# Select 4 additional centers sequentially
# ------------------------------------------------------------

for novel_step in range(5):

    # --------------------------------------------------------
    # Prevent selecting the exact same dataset point twice.
    # --------------------------------------------------------

    sampling_weights = (
        min_squared_distances.copy()
    )

    if selected_indices:

        sampling_weights[
            list(selected_indices)
        ] = 0.0


    total_weight = (
        sampling_weights.sum()
    )


    # --------------------------------------------------------
    # Degenerate case:
    # If every point has zero distance to an existing center,
    # select a random unused point.
    # --------------------------------------------------------

    if total_weight <= 0:

        available_indices = np.array(
            [
                i
                for i in range(len(X_all))
                if i not in selected_indices
            ],
            dtype=np.int64
        )

        selected_idx = rng.choice(
            available_indices
        )

    else:

        probabilities = (
            sampling_weights /
            total_weight
        )

        selected_idx = rng.choice(
            len(X_all),
            p=probabilities
        )


    selected_indices.add(
        int(selected_idx)
    )


    # --------------------------------------------------------
    # Add selected point as the next center.
    # --------------------------------------------------------

    new_center = X_all[
        selected_idx
    ]


    initial_centers = np.vstack([
        initial_centers,
        new_center
    ])


    # --------------------------------------------------------
    # Update each sample's distance to its nearest center.
    #
    # This includes:
    #
    #   10 known centers
    #   + all novel centers selected so far
    # --------------------------------------------------------

    new_squared_distances = np.sum(
        (
            X_all
            -
            new_center
        ) ** 2,
        axis=1
    )


    min_squared_distances = np.minimum(
        min_squared_distances,
        new_squared_distances
    )


    print(
        f"Novel center {novel_step + 1}/4 "
        f"selected from dataset row {selected_idx}"
    )


# ============================================================
# 4. Verify 14 initial centers
# ============================================================




print(
    "\nInitial centers shape:",
    initial_centers.shape
)
assert initial_centers.shape == (
    14,
    X_all.shape[1]
)
print(
    "✓ 10 known centers + "
    "4 K-Means++ centers created"
)


# ============================================================
# 5. Semi-supervised K-Means iterations
#
# ALL 14000 samples participate.
#
# Labeled samples are constrained:
#
#     labeled known sample
#          ↓
#     its true known-class cluster
#
# Unlabeled samples are assigned normally to the nearest
# of the 14 centers.
# ============================================================

print("\n" + "-" * 90)
print("STEP 3 — SEMI-SUPERVISED K-MEANS ITERATIONS")
print("-" * 90)


centers = initial_centers.copy()

MAX_ITER = 300
TOL = 1e-6


for iteration in range(MAX_ITER):

    old_centers = centers.copy()


    # --------------------------------------------------------
    # Distance from ALL 14000 samples to ALL 14 centers
    # --------------------------------------------------------

    distances = np.sum(
        (
            X_all[:, None, :]
            -
            centers[None, :, :]
        ) ** 2,
        axis=2
    )


    # --------------------------------------------------------
    # Standard nearest-center assignment
    # --------------------------------------------------------

    assignments = np.argmin(
        distances,
        axis=1
    )


    # --------------------------------------------------------
    # Semi-supervised constraint
    #
    # The 7000 labeled samples MUST remain assigned to their
    # true known class.
    # --------------------------------------------------------

    assignments[
        known_indices
    ] = y_all[
        known_indices
    ]


    # --------------------------------------------------------
    # Recalculate all 14 centers using ALL 14000 samples
    # --------------------------------------------------------

    new_centers = np.zeros_like(
        centers
    )


    for cluster_id in range(14):

        cluster_mask = (
            assignments == cluster_id
        )


        if np.any(cluster_mask):

            new_centers[
                cluster_id
            ] = X_all[
                cluster_mask
            ].mean(axis=0)

        else:

            # Empty cluster:
            # retain its previous center.
            new_centers[
                cluster_id
            ] = centers[
                cluster_id
            ]


    centers = new_centers


    # --------------------------------------------------------
    # Maximum center movement
    # --------------------------------------------------------

    center_shift = np.max(
        np.linalg.norm(
            centers - old_centers,
            axis=1
        )
    )


    if (
        iteration == 0
        or
        (iteration + 1) % 10 == 0
    ):

        print(
            f"Iteration {iteration + 1:3d} | "
            f"Max center shift = {center_shift:.10f}"
        )


    if center_shift < TOL:

        print(
            f"\n✓ Converged at iteration "
            f"{iteration + 1}"
        )

        break

else:

    print(
        f"\nReached maximum iterations: "
        f"{MAX_ITER}"
    )


# ============================================================
# 6. Final assignment using final centers
# ============================================================

final_distances = np.sum(
    (
        X_all[:, None, :]
        -
        centers[None, :, :]
    ) ** 2,
    axis=2
)


final_assignments = np.argmin(
    final_distances,
    axis=1
)


# Again enforce the known-label constraint
final_assignments[
    known_indices
] = y_all[
    known_indices
]


# ============================================================
# 7. EVALUATION — UNLABELED 7000 ONLY
# ============================================================

pred_unlabeled = (
    final_assignments[
        unknown_indices
    ]
)

true_unlabeled = (
    y_all[
        unknown_indices
    ]
)


# ============================================================
# 8. Hungarian matching
#
# IMPORTANT:
# The Hungarian mapping is calculated ONLY using the 7000
# evaluation samples.
# ============================================================

contingency = np.zeros(
    (14, 14),
    dtype=np.int64
)


for predicted_cluster, true_class in zip(
    pred_unlabeled,
    true_unlabeled
):

    contingency[
        predicted_cluster,
        true_class
    ] += 1


row_ind, col_ind = (
    linear_sum_assignment(
        -contingency
    )
)


# cluster ID -> true class ID
cluster_to_class = {
    cluster_id: true_class_id
    for cluster_id, true_class_id
    in zip(
        row_ind,
        col_ind
    )
}


mapped_predictions = np.array([
    cluster_to_class[
        cluster_id
    ]
    for cluster_id in pred_unlabeled
])


# ============================================================
# 9. Clustering accuracy
# ============================================================

accuracy = np.mean(
    mapped_predictions
    ==
    true_unlabeled
)


# ============================================================
# 10. ARI and NMI
# ============================================================

ari = adjusted_rand_score(
    true_unlabeled,
    pred_unlabeled
)


nmi = normalized_mutual_info_score(
    true_unlabeled,
    pred_unlabeled
)


# ============================================================
# 11. Silhouette
#
# Calculated on the 7000 unlabeled samples only.
# ============================================================

silhouette = silhouette_score(
    X_unlabeled,
    pred_unlabeled
)


# ============================================================
# 12. Print final metrics
# ============================================================

print("\n" + "=" * 90)
print("FINAL SEMI-SUPERVISED K-MEANS++ RESULTS")
print("=" * 90)

print(
    f"Samples used during clustering : 14000"
)

print(
    f"Labeled samples                : 7000"
)

print(
    f"Unlabeled samples              : 7000"
)

print(
    f"Number of clusters             : 14"
)

print(
    f"Feature dimensions             : "
    f"{X_all.shape[1]}"
)

print("\nEvaluation is performed ONLY on the 7000 unlabeled samples.")

print(
    f"\nClustering Accuracy : "
    f"{accuracy:.4f} "
    f"({accuracy * 100:.2f}%)"
)

print(
    f"ARI                 : "
    f"{ari:.4f}"
)

print(
    f"NMI                 : "
    f"{nmi:.4f}"
)

print(
    f"Silhouette          : "
    f"{silhouette:.4f}"
)


# ============================================================
# 13. Class-wise accuracy
# ============================================================

print("\n" + "-" * 90)
print("CLASS-WISE ACCURACY — UNLABELED 7000 ONLY")
print("-" * 90)


classwise_results = []


for class_id, class_name in enumerate(
    CLASS_ORDER
):

    mask = (
        true_unlabeled == class_id
    )

    total = int(
        mask.sum()
    )


    if total == 0:
        continue


    correct = int(
        np.sum(
            mapped_predictions[mask]
            == class_id
        )
    )


    class_accuracy = (
        correct / total
    )


    classwise_results.append({
        "Class": class_name,
        "Correct": correct,
        "Total": total,
        "Accuracy": class_accuracy
    })


    print(
        f"{class_name:20s} "
        f"{correct:4d}/{total:4d} "
        f"({class_accuracy * 100:6.2f}%)"
    )


classwise_df = pd.DataFrame(
    classwise_results
)


# ============================================================
# 14. CONFUSION MATRIX
#
# Rows    = true classes
# Columns = predicted classes after Hungarian matching
#
# ONLY the 7000 unlabeled samples are included.
# ============================================================

cm = confusion_matrix(
    true_unlabeled,
    mapped_predictions,
    labels=np.arange(14)
)


print("\n" + "=" * 90)
print("CONFUSION MATRIX — UNLABELED 7000 ONLY")
print("=" * 90)

print(
    pd.DataFrame(
        cm,
        index=CLASS_ORDER,
        columns=CLASS_ORDER
    )
)


# ============================================================
# 15. Plot confusion matrix
# ============================================================

fig, ax = plt.subplots(
    figsize=(13, 11)
)


im = ax.imshow(
    cm,
    interpolation="nearest"
)


ax.set_title(
    "Semi-Supervised K-Means++\n"
    "Confusion Matrix — Unlabeled 7000 Samples"
)


ax.set_xlabel(
    "Predicted Class"
)

ax.set_ylabel(
    "True Class"
)


ax.set_xticks(
    np.arange(14)
)

ax.set_yticks(
    np.arange(14)
)


ax.set_xticklabels(
    CLASS_ORDER,
    rotation=90
)

ax.set_yticklabels(
    CLASS_ORDER
)


# ------------------------------------------------------------
# Write count inside each cell
# ------------------------------------------------------------

threshold = (
    cm.max() / 2.0
)


for i in range(14):

    for j in range(14):

        ax.text(
            j,
            i,
            str(cm[i, j]),
            ha="center",
            va="center",
            color=(
                "white"
                if cm[i, j] > threshold
                else "black"
            ),
            fontsize=8
        )


fig.colorbar(
    im,
    ax=ax,
    fraction=0.046,
    pad=0.04,
    label="Number of samples"
)


plt.tight_layout()
plt.show()
