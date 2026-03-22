file structure
project/
│
├── original/                          # YOUR EXISTING DATA (read-only)
│   ├── Train/
│   │   ├── images/                    # .jpeg original full-res images
│   │   ├── json/                      # .json LabelMe polygon annotations
│   │   └── masks/                     # color-coded PNG masks (red/green/yellow)
│   └── Test/
│       ├── images/
│       ├── json/
│       └── masks/
│
├── data/
│   ├── converted/                     # Step 1: JSON → integer class masks
│   │   ├── Train/
│   │   │   ├── images/                # copied full-res images
│   │   │   └── masks/                 # grayscale masks (0,1,2,3 pixel values)
│   │   └── Test/
│   │       ├── images/
│   │       └── masks/
│   │
│   ├── patches/                       # Step 2: 512x512 sliding window crops
│   │   ├── Train/
│   │   │   ├── images/
│   │   │   └── masks/
│   │   └── Test/
│   │       ├── images/
│   │       └── masks/
│   │
│   └── final/                         # Step 3: train/val/test splits
│       ├── train/                     # ~5121 patches
│       │   ├── images/
│       │   └── masks/
│       ├── val/                       # ~570 patches
│       │   ├── images/
│       │   └── masks/
│       └── test/                      # ~338 patches (from original Test/)
│           ├── images/
│           └── masks/
│
├── models/
│   ├── eem.py
│   ├── ffm.py
│   └── segformer_eem_ffm.py
├── datasets/
│   └── corrosion_dataset.py
├── losses/
│   └── focal_loss.py
├── scripts/
│   ├── step1_convert_json_to_masks.py
│   ├── step2_slice_patches.py
│   └── step3_make_splits.py
├── checkpoints/
├── train.py
├── evaluate.py
└── utils.py



how to run 
download dataset and import it in proper file structure in original folder. then run scripts step 1 to 3 then run train. 