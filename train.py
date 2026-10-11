## Training

# ============================================================
# CONTRASTIVE LEARNING
# ============================================================

import os
import random
import time

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader


# ============================================================
# REPRODUCIBILITY
# ============================================================

TRAIN_SEED = 42

random.seed(TRAIN_SEED)
np.random.seed(TRAIN_SEED)
torch.manual_seed(TRAIN_SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(TRAIN_SEED)


# Optional CUDA reproducibility
if torch.cuda.is_available():

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_DIM = 768

HIDDEN_DIM = 512

EMBEDDING_DIM = 128

BATCH_SIZE = 500

EPOCHS = 100

LEARNING_RATE = 1e-3

WEIGHT_DECAY = 1e-5

TEMPERATURE = 0.1

LAMBDA = 0.6


print("\nConfiguration")
print("-" * 60)

print("Input dimension :", INPUT_DIM)
print("Hidden dimension:", HIDDEN_DIM)
print("Embedding dim   :", EMBEDDING_DIM)
print("Batch size      :", BATCH_SIZE)
print("Epochs          :", EPOCHS)
print("Learning rate   :", LEARNING_RATE)
print("Temperature     :", TEMPERATURE)
print("Lambda          :", LAMBDA)

# ============================================================
# PREPARE ALL 14,000 ORIGINAL + AUGMENTED FEATURES
# ============================================================

X_real_all = df_real[
    FEATURE_COLUMNS
].to_numpy(
    dtype=np.float32
)

X_aug_all = df_aug[
    FEATURE_COLUMNS
].to_numpy(
    dtype=np.float32
)


# ============================================================
# TEXT LABELS
# ============================================================

labels_text_all = (
    df_real["label"]
    .astype(str)
    .to_numpy()
)


# ============================================================
# FIXED CLASS -> INTEGER MAPPING
#
# IMPORTANT:
# IDs 0-9   = known classes
# IDs 10-13 = novel classes
# ============================================================

label_to_int = {
    class_name: class_id
    for class_id, class_name
    in enumerate(CLASS_ORDER)
}


labels_all = np.array(
    [
        label_to_int[label]
        for label in labels_text_all
    ],
    dtype=np.int64
)


# ============================================================
# CRITICAL:
# USE THE MASTER 7,000 LABELED INDICES
#
# DO NOT DEFINE THIS USING:
#
#     label in KNOWN_CLASSES
#
# because that would mark all 10,000 known-class samples
# as supervised.
# ============================================================

known_mask_all = np.zeros(
    EXPECTED_TOTAL_SAMPLES,
    dtype=bool
)

known_mask_all[
    known_indices
] = True


# ============================================================
# SUPERVISED LABELS
#
# Labeled known samples:
#     true class ID
#
# Unlabeled samples:
#     -1
# ============================================================

supervised_labels_all = np.where(
    known_mask_all,
    labels_all,
    -1
).astype(
    np.int64
)


# ============================================================
# CRITICAL VERIFICATION
# ============================================================

assert X_real_all.shape == (
    14000,
    768
)

assert X_aug_all.shape == (
    14000,
    768
)

assert labels_all.shape == (
    14000,
)

assert supervised_labels_all.shape == (
    14000,
)


# The labeled indices must be exactly the master split
assert np.array_equal(
    np.where(known_mask_all)[0],
    np.sort(known_indices)
)


# Real/augmented alignment
assert np.array_equal(
    df_real["filepath"]
    .astype(str)
    .to_numpy(),

    df_aug["filepath"]
    .astype(str)
    .to_numpy()
)


print("=" * 70)
print("CONTRASTIVE TRAINING DATA")
print("=" * 70)

print("Real features      :", X_real_all.shape)
print("Augmented features :", X_aug_all.shape)

print("Labeled samples    :", known_mask_all.sum())
print("Unlabeled samples  :", (~known_mask_all).sum())

print(
    "\n✓ EXACTLY 7,000 samples have supervised labels."
)

print(
    "✓ EXACTLY 7,000 samples are treated as unlabeled."
)

# ============================================================
# PYTORCH DATASET
# ============================================================

class ContrastiveMusicDataset(Dataset):

    def __init__(
        self,
        real_features,
        augmented_features,
        supervised_labels,
        known_mask
    ):

        self.real_features = torch.from_numpy(
            real_features
        )

        self.augmented_features = torch.from_numpy(
            augmented_features
        )

        self.supervised_labels = torch.from_numpy(
            supervised_labels
        ).long()

        self.known_mask = torch.from_numpy(
            known_mask.astype(np.bool_)
        )


    def __len__(self):

        return len(
            self.real_features
        )


    def __getitem__(self, index):

        return {
            "real": self.real_features[index],

            "augmented":
                self.augmented_features[index],

            "label":
                self.supervised_labels[index],

            "known":
                self.known_mask[index],

            "index":
                index
        }


dataset = ContrastiveMusicDataset(
    real_features=X_real_all,
    augmented_features=X_aug_all,
    supervised_labels=supervised_labels_all,
    known_mask=known_mask_all
)


assert len(dataset) == 14000

print("Dataset size:", len(dataset))

# ============================================================
# DATA LOADER
# ============================================================

train_loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    drop_last=False,
    num_workers=2,
    pin_memory=torch.cuda.is_available()
)


print(
    "Number of batches per epoch:",
    len(train_loader)
)

# ============================================================
# DEEP CONTRASTIVE PROJECTION NETWORK
# ============================================================

class ContrastiveNetwork(nn.Module):

    def __init__(
        self,
        input_dim=768,
        embedding_dim=128
    ):
        super().__init__()

        # ----------------------------------------------------
        # Input projection
        # 768 -> 1024
        # ----------------------------------------------------
        self.input_block = nn.Sequential(

            nn.Linear(
                input_dim,
                1024
            ),

            nn.BatchNorm1d(
                1024
            ),

            nn.GELU(),

            nn.Dropout(
                p=0.15
            )
        )

        # ----------------------------------------------------
        # Hidden block 1
        # 1024 -> 512
        # ----------------------------------------------------
        self.block1 = nn.Sequential(

            nn.Linear(
                1024,
                512
            ),

            nn.BatchNorm1d(
                512
            ),

            nn.GELU(),

            nn.Dropout(
                p=0.15
            )
        )

        # ----------------------------------------------------
        # Hidden block 2
        # 512 -> 256
        # ----------------------------------------------------
        self.block2 = nn.Sequential(

            nn.Linear(
                512,
                256
            ),

            nn.BatchNorm1d(
                256
            ),

            nn.GELU(),

            nn.Dropout(
                p=0.10
            )
        )

        # ----------------------------------------------------
        # Projection block
        # 256 -> 128
        # ----------------------------------------------------
        self.projection = nn.Sequential(

            nn.Linear(
                256,
                embedding_dim
            )
        )


    def forward(self, x):

        # ----------------------------------------------------
        # Feature transformation
        # ----------------------------------------------------
        x = self.input_block(x)

        x = self.block1(x)

        h = self.block2(x)

        # ----------------------------------------------------
        # Final contrastive projection
        # ----------------------------------------------------
        z = self.projection(h)

        # ----------------------------------------------------
        # L2 normalization
        # ----------------------------------------------------
        z = F.normalize(
            z,
            p=2,
            dim=1
        )

        return z, h


# ============================================================
# CREATE MODEL
# ============================================================

model = ContrastiveNetwork(
    input_dim=INPUT_DIM,
    embedding_dim=EMBEDDING_DIM
).to(device)


print(model)


# ============================================================
# PARAMETER COUNT
# ============================================================

total_params = sum(
    p.numel()
    for p in model.parameters()
)

trainable_params = sum(
    p.numel()
    for p in model.parameters()
    if p.requires_grad
)

print("\nTotal parameters    :", total_params)
print("Trainable parameters:", trainable_params)

# ============================================================
# OPTIMIZER
# ============================================================

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)


print(optimizer)

# ============================================================
# UNSUPERVISED CONTRASTIVE LOSS
# ============================================================

def unsupervised_contrastive_loss(
    z_real,
    z_aug,
    temperature=0.07
):
    """
    NT-Xent / InfoNCE-style loss.

    Every real sample and its corresponding augmented
    sample form a positive pair.

    All other samples are negatives.

    This loss uses ALL samples, including unlabeled samples.
    That is appropriate for a transductive GCD experiment.
    """

    batch_size = z_real.size(0)


    # --------------------------------------------------------
    # Two views
    # --------------------------------------------------------

    z = torch.cat(
        [
            z_real,
            z_aug
        ],
        dim=0
    )


    # --------------------------------------------------------
    # Similarity matrix
    # --------------------------------------------------------

    logits = torch.matmul(
        z,
        z.T
    )

    logits = (
        logits /
        temperature
    )


    # --------------------------------------------------------
    # Remove self similarity
    # --------------------------------------------------------

    total_views = (
        2 * batch_size
    )

    self_mask = torch.eye(
        total_views,
        device=z.device,
        dtype=torch.bool
    )


    logits = logits.masked_fill(
        self_mask,
        -torch.inf
    )


    # --------------------------------------------------------
    # Positive indices
    # --------------------------------------------------------

    positive_indices = torch.cat(
        [
            torch.arange(
                batch_size,
                2 * batch_size,
                device=z.device
            ),

            torch.arange(
                0,
                batch_size,
                device=z.device
            )
        ]
    )


    # --------------------------------------------------------
    # InfoNCE
    # --------------------------------------------------------

    loss = F.cross_entropy(
        logits,
        positive_indices
    )


    return loss

# ============================================================
# SUPERVISED CONTRASTIVE LOSS
# ============================================================

def supervised_contrastive_loss(
    z_real,
    z_aug,
    labels,
    temperature=0.07
):

    # --------------------------------------------------------
    # Only labeled samples
    # --------------------------------------------------------

    known_mask = (
        labels >= 0
    )


    n_known = known_mask.sum()


    if n_known < 2:

        return torch.tensor(
            0.0,
            device=z_real.device,
            dtype=z_real.dtype,
            requires_grad=True
        )


    z_real_known = z_real[
        known_mask
    ]

    z_aug_known = z_aug[
        known_mask
    ]

    labels_known = labels[
        known_mask
    ]


    # --------------------------------------------------------
    # Two views
    # --------------------------------------------------------

    z = torch.cat(
        [
            z_real_known,
            z_aug_known
        ],
        dim=0
    )

    labels_two_views = torch.cat(
        [
            labels_known,
            labels_known
        ],
        dim=0
    )


    # --------------------------------------------------------
    # Similarity
    # --------------------------------------------------------

    logits = torch.matmul(
        z,
        z.T
    ) / temperature


    n_views = z.size(0)


    # --------------------------------------------------------
    # Self mask
    # --------------------------------------------------------

    self_mask = torch.eye(
        n_views,
        device=z.device,
        dtype=torch.bool
    )


    # --------------------------------------------------------
    # Positive mask
    # --------------------------------------------------------

    positive_mask = (
        labels_two_views.unsqueeze(0)
        ==
        labels_two_views.unsqueeze(1)
    )

    positive_mask = (
        positive_mask
        &
        ~self_mask
    )


    # --------------------------------------------------------
    # Remove self from denominator
    # --------------------------------------------------------

    logits_without_self = (
        logits.masked_fill(
            self_mask,
            -torch.inf
        )
    )


    log_denominator = torch.logsumexp(
        logits_without_self,
        dim=1,
        keepdim=True
    )


    log_prob = (
        logits -
        log_denominator
    )


    # --------------------------------------------------------
    # NUMERICAL STABILITY
    #
    # Avoid:
    #
    # 0 * (-inf) = NaN
    # --------------------------------------------------------

    safe_log_prob = torch.where(
        positive_mask,
        log_prob,
        torch.zeros_like(log_prob)
    )


    positive_count = (
        positive_mask.sum(
            dim=1
        )
    )


    valid_anchor_mask = (
        positive_count > 0
    )


    if valid_anchor_mask.sum() == 0:

        return torch.tensor(
            0.0,
            device=z.device,
            dtype=z.dtype,
            requires_grad=True
        )


    mean_log_prob_positive = (
        safe_log_prob.sum(
            dim=1
        )
        /
        positive_count.clamp_min(1)
    )


    loss = -mean_log_prob_positive[
        valid_anchor_mask
    ].mean()


    return loss

# ============================================================
# COMBINED CONTRASTIVE LOSS
# ============================================================

def combined_contrastive_loss(
    z_real,
    z_aug,
    labels,
    temperature=0.07,
    lambda_value=0.5
):

    loss_unsup = (
        unsupervised_contrastive_loss(
            z_real=z_real,
            z_aug=z_aug,
            temperature=temperature
        )
    )


    loss_sup = (
        supervised_contrastive_loss(
            z_real=z_real,
            z_aug=z_aug,
            labels=labels,
            temperature=temperature
        )
    )


    loss = (
        lambda_value * loss_sup
        +
        (1.0 - lambda_value)
        * loss_unsup
    )


    return (
        loss,
        loss_sup,
        loss_unsup
    )

# ============================================================
# LOSS SANITY CHECK
# ============================================================

model.eval()

batch = next(
    iter(train_loader)
)


real = batch["real"].to(
    device,
    non_blocking=True
)

augmented = batch["augmented"].to(
    device,
    non_blocking=True
)

labels = batch["label"].to(
    device,
    non_blocking=True
)


with torch.no_grad():

    z_real = model(real)[0]

    z_aug = model(
        augmented
    )[0]


loss, loss_sup, loss_unsup = (
    combined_contrastive_loss(
        z_real,
        z_aug,
        labels,
        temperature=TEMPERATURE,
        lambda_value=LAMBDA
    )
)


print("=" * 70)
print("LOSS SANITY CHECK")
print("=" * 70)

print("Batch size       :", real.shape[0])

print("Real shape       :", real.shape)

print(
    "Augmented shape  :",
    augmented.shape
)

print(
    "Known in batch   :",
    (labels >= 0).sum().item()
)

print(
    "Unknown in batch :",
    (labels < 0).sum().item()
)

print(
    "Supervised loss  :",
    loss_sup.item()
)

print(
    "Unsupervised loss:",
    loss_unsup.item()
)

print(
    "Combined loss    :",
    loss.item()
)


assert torch.isfinite(loss)
assert torch.isfinite(loss_sup)
assert torch.isfinite(loss_unsup)

# ============================================================
# CONTRASTIVE TRAINING
# ============================================================

history = {
    "total_loss": [],
    "supervised_loss": [],
    "unsupervised_loss": []
}


print("=" * 70)
print("STARTING CONTRASTIVE TRAINING")
print("=" * 70)


for epoch in range(
    1,
    EPOCHS + 1
):

    model.train()


    epoch_total_loss = 0.0
    epoch_sup_loss = 0.0
    epoch_unsup_loss = 0.0

    n_batches = 0

    start_time = time.time()


    for batch in train_loader:

        real = batch["real"].to(
            device,
            non_blocking=True
        )

        augmented = batch["augmented"].to(
            device,
            non_blocking=True
        )

        labels = batch["label"].to(
            device,
            non_blocking=True
        )


        # ----------------------------------------------------
        # Forward
        # ----------------------------------------------------

        z_real = model(real)[0]

        z_aug = model(
            augmented
        )[0]


        # ----------------------------------------------------
        # Loss
        # ----------------------------------------------------

        loss, loss_sup, loss_unsup = (
            combined_contrastive_loss(
                z_real=z_real,
                z_aug=z_aug,
                labels=labels,
                temperature=TEMPERATURE,
                lambda_value=LAMBDA
            )
        )


        # ----------------------------------------------------
        # Safety check
        # ----------------------------------------------------

        if not torch.isfinite(loss):

            raise RuntimeError(
                f"Non-finite loss at epoch {epoch}"
            )


        # ----------------------------------------------------
        # Backpropagation
        # ----------------------------------------------------

        optimizer.zero_grad(
            set_to_none=True
        )

        loss.backward()

        optimizer.step()


        # ----------------------------------------------------
        # Accumulate
        # ----------------------------------------------------

        epoch_total_loss += (
            loss.item()
        )

        epoch_sup_loss += (
            loss_sup.item()
        )

        epoch_unsup_loss += (
            loss_unsup.item()
        )

        n_batches += 1


    # --------------------------------------------------------
    # Average losses
    # --------------------------------------------------------

    avg_total_loss = (
        epoch_total_loss /
        n_batches
    )

    avg_sup_loss = (
        epoch_sup_loss /
        n_batches
    )

    avg_unsup_loss = (
        epoch_unsup_loss /
        n_batches
    )


    history["total_loss"].append(
        avg_total_loss
    )

    history["supervised_loss"].append(
        avg_sup_loss
    )

    history["unsupervised_loss"].append(
        avg_unsup_loss
    )


    epoch_time = (
        time.time() -
        start_time
    )


    print(
        f"Epoch [{epoch:03d}/{EPOCHS}] | "
        f"Total: {avg_total_loss:.4f} | "
        f"Sup: {avg_sup_loss:.4f} | "
        f"Unsup: {avg_unsup_loss:.4f} | "
        f"Time: {epoch_time:.2f}s"
    )

# # ============================================================
# # SAVE MODEL
# # ============================================================

# MODEL_PATH = (
#     "/content/"
#     "mert_contrastive_network_corrected.pth"
# )


# torch.save(
#     {
#         "model_state_dict":
#             model.state_dict(),

#         "input_dim":
#             INPUT_DIM,

#         "hidden_dim":
#             HIDDEN_DIM,

#         "embedding_dim":
#             EMBEDDING_DIM,

#         "temperature":
#             TEMPERATURE,

#         "lambda":
#             LAMBDA,

#         "train_seed":
#             TRAIN_SEED,

#         "split_seed":
#             SEED,

#         "known_classes":
#             KNOWN_CLASSES,

#         "unknown_classes":
#             UNKNOWN_CLASSES,

#         "class_order":
#             CLASS_ORDER,

#         "history":
#             history
#     },
#     MODEL_PATH
# )


# print("Corrected model saved to:")
# print(MODEL_PATH)

# ============================================================
# TRAINING LOSS CURVES
# ============================================================

import matplotlib.pyplot as plt


epochs_axis = np.arange(
    1,
    EPOCHS + 1
)


plt.figure(
    figsize=(10, 5)
)

plt.plot(
    epochs_axis,
    history["total_loss"],
    label="Total Loss"
)

plt.plot(
    epochs_axis,
    history["supervised_loss"],
    label="Supervised Contrastive Loss"
)

plt.plot(
    epochs_axis,
    history["unsupervised_loss"],
    label="Unsupervised Contrastive Loss"
)

plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.title(
    "Corrected Contrastive Training Loss"
)

plt.legend()

plt.grid(True)

plt.show()



# ============================================================
# GENERATE LEARNED EMBEDDINGS
# ============================================================

model.eval()


all_real_tensor = torch.from_numpy(
    X_real_all
)


embedding_batches = []


with torch.inference_mode():

    for start in range(
        0,
        len(all_real_tensor),
        BATCH_SIZE
    ):

        batch_real = (
            all_real_tensor[
                start:start + BATCH_SIZE
            ]
            .to(
                device,
                non_blocking=True
            )
        )


        z = model(
            batch_real
        )[1]


        embedding_batches.append(
            z.cpu().numpy()
        )


X_learned = np.concatenate(
    embedding_batches,
    axis=0
)
print(z.shape)

print(
    "Learned embedding shape:",
    X_learned.shape
)

# ============================================================
# FINAL EMBEDDING VERIFICATION
# ============================================================



embedding_norms = np.linalg.norm(
    X_learned,
    axis=1
)


print(
    "Minimum embedding norm:",
    embedding_norms.min()
)

print(
    "Maximum embedding norm:",
    embedding_norms.max()
)

print(
    "Mean embedding norm:",
    embedding_norms.mean()
)



print("\n✓ Learned embeddings verified.")

# ============================================================
# VERIFY MASTER GCD SPLIT FOR EVALUATION
#
# IMPORTANT:
# DO NOT CREATE A NEW RANDOM SPLIT HERE.
#
# Cell 7 already created:
#     known_indices
#     unknown_indices
#
# We reuse those exact indices.
# ============================================================

print("=" * 70)
print("VERIFYING MASTER GCD SPLIT")
print("=" * 70)



# ------------------------------------------------------------
# Verify no overlap
# ------------------------------------------------------------

assert len(
    np.intersect1d(
        known_indices,
        unknown_indices
    )
) == 0


# ------------------------------------------------------------
# Verify complete coverage
# ------------------------------------------------------------

assert len(
    np.union1d(
        known_indices,
        unknown_indices
    )
) == 14000


# ------------------------------------------------------------
# Create master integer labels
# ------------------------------------------------------------

label_to_int = {
    class_name: class_id
    for class_id, class_name
    in enumerate(CLASS_ORDER)
}

int_to_label = {
    class_id: class_name
    for class_name, class_id
    in label_to_int.items()
}


y_split = np.array(
    [
        label_to_int[str(label)]
        for label in df_real["label"]
    ],
    dtype=np.int64
)


assert y_split.shape == (
    14000,
)


# ------------------------------------------------------------
# Create boolean labeled mask
# ------------------------------------------------------------

known_mask_all = np.zeros(
    14000,
    dtype=bool
)

known_mask_all[
    known_indices
] = True




# ------------------------------------------------------------
# Verify every known class has 700 labels
# ------------------------------------------------------------

print("\nPer-class split:")

for class_id, class_name in enumerate(
    CLASS_ORDER
):

    class_mask = (
        y_split == class_id
    )

    labeled_count = np.sum(
        known_mask_all &
        class_mask
    )

    unlabeled_count = np.sum(
        (~known_mask_all) &
        class_mask
    )


    print(
        f"{class_name:20s} "
        f"Labeled={labeled_count:4d} "
        f"Unlabeled={unlabeled_count:4d}"
    )


    if class_name in KNOWN_CLASSES:

        assert labeled_count == 700
        assert unlabeled_count == 300

    else:

        assert labeled_count == 0
        assert unlabeled_count == 1000


print("\n✓ Master GCD split verified.")
