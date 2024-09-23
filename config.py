
# model weight file list
MODEL_ASSETS = {
    "C4-fine-tuned": [
        "assets/c4_vit32_vtt_b128_ep5.pth",
        "assets/2024_02_06_case3_ViT_base16.bin",
    ],
    "ViP-fine-tuned": [
        "assets/clip_vip_msrvtt_reproduce_best.pt",
        "assets/vip_vit32_case3_b128_ep5.pth",
        "assets/2023_12_14_case3_best.pt",
        "assets/clipvip_zeroshot_base_32.pt",
        "assets/2024_02_06_case3_ViT-B16.pt",
    ],
}

MODEL_DOWNLOAD_ID = {
    "assets/c4_vit32_vtt_b128_ep5.pth": "1SZI2mQJxE9yewv3WpRNlAkxMVI6OM0nc",
    "assets/2023_12_14_case3_best.pt": "1Ys8we-ECtzBEa9WSUXFqRgP0vcICk7E9",
    "assets/clip_vip_msrvtt_reproduce_best.pt": "1fXofnihp5ekPGDK6Ye9RIstApQ61tfEI",
    "assets/vip_vit32_case3_b128_ep5.pth": "1Ys8we-ECtzBEa9WSUXFqRgP0vcICk7E9",
    "assets/clipvip_zeroshot_base_32.pt": "1xQPCeR1sbL0oM5jYYvlaeWsulPxgnot2",
    "assets/2024_02_06_case3_ViT-B16.pt": "1rBeRQMLJ97_ZbCRmY9kGxFYp6pJQOEKT",
    "assets/2024_02_06_case3_ViT_base16.bin": "1SzxQBir3BMB84KiDCiJqro6EPUV8S0UE",
}

GDRIVE_PREFIX = "https://drive.google.com/uc?export=download&id="

PROMPT_CATEGORY = [
    "Falldown",
    "Fire",
    "Smoke",
    "Smoking",
    "Violence",
    "Dangerous Work",
    "Accident",
    "Constriction",
]

PROMPT_TEXT = {
    0: [
        "A person is lying on the ground",
        "A person is lying on the floor",
        "A person is sprawled out on the ground",
        "A person is prostrate on the ground",
        "A person is collapse on the ground.",
        "A person is topple on the ground",
    ],
    1: [
        "A fire is burning",
        "Flames are burning",
        "Flames erupted on the ground",
        "Embers are burning on the ground",
    ],
    2: [
        "There appears to be smoke rising",
        "Smoke is rising",
        "Vapor is coming up",
        "faint presence of smoke or steam rising",
        "there is a subtle presence of smoke or steam",
        "Smoke is rising from the fire",
    ],
    3: [
        "smoking people",
        "smoking a cigarette",
        "holding a cigarette in ones hand",
        "blowing smoke",
        "taking a puff",
        "Cigarette smoke is coming out of mouth",
        "person with cigarette in ones mouth",
        "it looks like someone is smoking",
    ],
    4: [
        "violence with kicking and punching",
        "physical confrontation between people",
        "violence broke out",
    ],
    5: [
        "workers are working near heavy objects",
        "workers are getting too close to heavy objects",
    ],
    6: [
        "serious accident occurred",
        "heavy objects are fallen",
    ],
    7: [
        "people are concentrated in small places",
        "people are crowded in small place",
    ],

}


PROMPT_RANGE = {
    "file_1": {
        0: [
            0.255, 0.255, 0.255, 0.255, 0.255, 0.255,
        ],
        1: [
            0.22, 0.22, 0.22, 0.25,
        ],
        2: [
            0.23, 0.23, 0.23, 0.23, 0.23, 0.23,
        ],
        3: [
            0.23, 0.23, 0.23, 0.23, 0.23, 0.23, 0.23, 0.23,
        ],
        4: [
            0.245, 0.245, 0.245,
        ],
        5: [
            0.26, 0.26,
        ],
        6: [
            0.24, 0.255,
        ],
        7: [
            0.26, 0.26,
        ],
    },
}

MAX_WIDTH = 854
MAX_HEIGHT = 480

DEVICE = "cuda"
# DEFAULT_RTSP = "192.168.36.34/66/high"
DEFAULT_RTSP = ""
CAM_ID = "file_1"
