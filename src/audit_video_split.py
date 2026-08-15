from pathlib import Path
import re
import pandas as pd


# ============================================================
# PATHS
# ============================================================

METADATA_PATH = Path(
    r"datasets\video\prepared\metadata.csv"
)

RAW_DIR = Path(
    r"datasets\video\celebdf_raw"
)

TEST_LIST_PATH = (
    RAW_DIR / "List_of_testing_videos.txt"
)


# ============================================================
# LOAD METADATA
# ============================================================

metadata = pd.read_csv(
    METADATA_PATH
)


print(
    "CELEB-DF DATASET SPLIT AUDIT"
)

print(
    "============================"
)

print(
    "Total videos:",
    len(metadata)
)


# ============================================================
# BASIC SPLIT CHECK
# ============================================================

print(
    "\nSPLIT DISTRIBUTION"
)

print(
    metadata.groupby(
        ["split", "label_name"]
    ).size()
)


# ============================================================
# NORMALIZE RELATIVE PATH
# ============================================================

metadata["relative_path"] = (
    metadata["relative_path"]
    .astype(str)
    .str.replace(
        "\\",
        "/",
        regex=False
    )
)


# ============================================================
# SOURCE CATEGORY
# ============================================================

def get_source(path):

    path = str(path).lower()


    if path.startswith(
        "celeb-real/"
    ):

        return "Celeb-real"


    if path.startswith(
        "celeb-synthesis/"
    ):

        return "Celeb-synthesis"


    if path.startswith(
        "youtube-real/"
    ):

        return "YouTube-real"


    return "Unknown"


metadata["source_category"] = (
    metadata["relative_path"]
    .apply(get_source)
)


print(
    "\nSOURCE DISTRIBUTION BY SPLIT"
)

print(
    metadata.groupby(
        [
            "split",
            "source_category"
        ]
    ).size()
)


# ============================================================
# IDENTITY EXTRACTION
# ============================================================

def extract_identities(path):

    path = str(path)


    # --------------------------------------------------------
    # Celeb-real
    #
    # Example:
    # celeb-real/id0_0000.mp4
    # --------------------------------------------------------

    if path.lower().startswith(
        "celeb-real/"
    ):

        filename = Path(
            path
        ).name


        match = re.match(
            r"(id\d+)_\d+\.mp4$",
            filename,
            re.IGNORECASE
        )


        if match:

            return [
                match.group(1).lower()
            ]


        return []


    # --------------------------------------------------------
    # Celeb-synthesis
    #
    # Example:
    # celeb-synthesis/id0_id16_0000.mp4
    # --------------------------------------------------------

    if path.lower().startswith(
        "celeb-synthesis/"
    ):

        filename = Path(
            path
        ).name


        match = re.match(
            r"(id\d+)_(id\d+)_\d+\.mp4$",
            filename,
            re.IGNORECASE
        )


        if match:

            return [
                match.group(1).lower(),
                match.group(2).lower()
            ]


        return []


    # --------------------------------------------------------
    # YouTube-real
    # --------------------------------------------------------

    return []


metadata["identities"] = (
    metadata["relative_path"]
    .apply(
        extract_identities
    )
)


# ============================================================
# SOURCE COUNTS
# ============================================================

print(
    "\nSOURCE COUNTS"
)

print(
    "============="
)

print(
    metadata[
        "source_category"
    ].value_counts()
)


# ============================================================
# IDENTITY SETS
# ============================================================

identity_sets = {}


for split in [
    "train",
    "validation",
    "test"
]:

    rows = metadata[
        metadata["split"] == split
    ]


    identities = set()


    for identity_list in rows[
        "identities"
    ]:

        identities.update(
            identity_list
        )


    identity_sets[
        split
    ] = identities


    print(
        f"\n{split.upper()} IDENTITIES"
    )

    print(
        "Identity count:",
        len(identities)
    )

    print(
        sorted(
            identities,
            key=lambda x: int(
                x[2:]
            )
        )
    )


# ============================================================
# IDENTITY OVERLAP
# ============================================================

print(
    "\nIDENTITY OVERLAP"
)

print(
    "================"
)


train_val_overlap = (
    identity_sets["train"]
    &
    identity_sets["validation"]
)


train_test_overlap = (
    identity_sets["train"]
    &
    identity_sets["test"]
)


val_test_overlap = (
    identity_sets["validation"]
    &
    identity_sets["test"]
)


print(
    "Train ∩ Validation:",
    len(train_val_overlap)
)

print(
    "Train ∩ Test:",
    len(train_test_overlap)
)

print(
    "Validation ∩ Test:",
    len(val_test_overlap)
)


if train_val_overlap:

    print(
        "\nTrain/Validation overlapping identities:"
    )

    print(
        sorted(
            train_val_overlap
        )
    )


if train_test_overlap:

    print(
        "\nTrain/Test overlapping identities:"
    )

    print(
        sorted(
            train_test_overlap
        )
    )


if val_test_overlap:

    print(
        "\nValidation/Test overlapping identities:"
    )

    print(
        sorted(
            val_test_overlap
        )
    )


# ============================================================
# VIDEOS PER IDENTITY
# ============================================================

print(
    "\nVIDEOS PER IDENTITY"
)

print(
    "==================="
)


identity_records = []


for _, row in metadata.iterrows():

    for identity in row[
        "identities"
    ]:

        identity_records.append(
            {
                "identity":
                    identity,

                "split":
                    row["split"],

                "label":
                    row["label_name"],

                "relative_path":
                    row["relative_path"]
            }
        )


identity_df = pd.DataFrame(
    identity_records
)


if not identity_df.empty:

    counts = (
        identity_df
        .groupby(
            [
                "split",
                "identity"
            ]
        )
        .size()
        .unstack(
            fill_value=0
        )
    )


    print(
        counts.to_string()
    )


# ============================================================
# OFFICIAL TEST LIST
# ============================================================

print(
    "\nOFFICIAL TEST LIST AUDIT"
)

print(
    "========================"
)


if not TEST_LIST_PATH.exists():

    print(
        "ERROR: Official test list not found."
    )

else:

    lines = [
        line.strip()
        for line in TEST_LIST_PATH
        .read_text(
            encoding="utf-8"
        )
        .splitlines()
        if line.strip()
    ]


    official_records = []


    for line in lines:

        parts = line.split(
            maxsplit=1
        )


        if len(parts) != 2:

            continue


        label = int(
            parts[0]
        )


        path = (
            parts[1]
            .replace(
                "\\",
                "/"
            )
            .lower()
        )


        official_records.append(
            {
                "label":
                    label,

                "relative_path":
                    path
            }
        )


    official_df = pd.DataFrame(
        official_records
    )


    print(
        "Official test entries:",
        len(official_df)
    )


    print(
        "\nOfficial test labels:"
    )

    print(
        official_df[
            "label"
        ]
        .value_counts()
        .sort_index()
    )


    # --------------------------------------------------------
    # Prepared test paths
    # --------------------------------------------------------

    prepared_test = metadata[
        metadata["split"] == "test"
    ].copy()


    prepared_paths = set(
        prepared_test[
            "relative_path"
        ]
        .str.lower()
    )


    official_paths = set(
        official_df[
            "relative_path"
        ]
    )


    missing_from_prepared = (
        official_paths
        -
        prepared_paths
    )


    extra_in_prepared = (
        prepared_paths
        -
        official_paths
    )


    print(
        "\nOfficial paths missing from prepared test:",
        len(missing_from_prepared)
    )


    print(
        "Prepared test paths not in official list:",
        len(extra_in_prepared)
    )


    # --------------------------------------------------------
    # Label consistency
    #
    # Celeb-DF:
    # 0 = fake
    # 1 = real
    # --------------------------------------------------------

    official_lookup = dict(
        zip(
            official_df[
                "relative_path"
            ],
            official_df[
                "label"
            ]
        )
    )


    label_mismatches = []


    for _, row in prepared_test.iterrows():

        path = (
            str(
                row[
                    "relative_path"
                ]
            )
            .lower()
        )


        if path not in official_lookup:

            continue


        official_label = (
            official_lookup[path]
        )


        prepared_label = int(
            row["label"]
        )


        if (
            official_label
            !=
            prepared_label
        ):

            label_mismatches.append(
                {
                    "path":
                        path,

                    "official":
                        official_label,

                    "prepared":
                        prepared_label
                }
            )


    print(
        "Label mismatches:",
        len(label_mismatches)
    )


    if label_mismatches:

        print(
            "\nLabel mismatch examples:"
        )

        print(
            pd.DataFrame(
                label_mismatches
            )
            .head(10)
            .to_string(
                index=False
            )
        )


# ============================================================
# FINAL SUMMARY
# ============================================================

print(
    "\nAUDIT SUMMARY"
)

print(
    "============="
)

print(
    "Total videos:",
    len(metadata)
)


for split in [
    "train",
    "validation",
    "test"
]:

    print(
        f"{split.capitalize()}:",
        len(
            metadata[
                metadata["split"] == split
            ]
        )
    )


print(
    "\nIdentity overlap:"
)

print(
    "Train/Validation:",
    len(train_val_overlap)
)

print(
    "Train/Test:",
    len(train_test_overlap)
)

print(
    "Validation/Test:",
    len(val_test_overlap)
)


print(
    "\nAUDIT COMPLETED"
)