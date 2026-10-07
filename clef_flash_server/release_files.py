"""Pinned runtime files from Cloudflare/clef-flash revision 17f0b0ad64efb65d273590632833508766b2aae6."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ExpectedFile:
    path: str
    bytes: int
    etag: str


RELEASE_FILES = (
    ExpectedFile(
        "chat_template.jinja", 7756, "a585dec894e63da457d9440ec6aa7caa16d20860"
    ),
    ExpectedFile("config.json", 2832, "aff766f2ab6e0e48b6aa297265689b981bdd2042"),
    ExpectedFile(
        "generation_config.json", 116, "54056568766c020050fab57761fea673becbc131"
    ),
    ExpectedFile(
        "joint_head.safetensors",
        243538016,
        "19cdcec8c81dc9212be320fff47462ab342fbc1278be4368fb3da71241cf5ba0",
    ),
    ExpectedFile(
        "joint_head_config.json", 119, "52467b6c7f7d1a7e628b518eb7a6cd7e9bdb9a57"
    ),
    ExpectedFile(
        "model-00001-of-00004.safetensors",
        4942706120,
        "8b45a8e968141cdcc58fb71c9adfc258e2c77b5f062bc636c1fd5bc5d916b565",
    ),
    ExpectedFile(
        "model-00002-of-00004.safetensors",
        4987757928,
        "7590856c713eed844a2dcf48e6c43c4de165b788bc3f80e328311183cdbc7db8",
    ),
    ExpectedFile(
        "model-00003-of-00004.safetensors",
        4954810240,
        "e6eac2467952c33361ed7dcb3c7959d1086bbe57201cd3749c3d769fdc17fe63",
    ),
    ExpectedFile(
        "model-00004-of-00004.safetensors",
        3934446832,
        "9fcecc6556b39171238373a465f409794b7f821fb4cd1e6459e3a9c0fe317af7",
    ),
    ExpectedFile(
        "model.safetensors.index.json",
        69253,
        "778b7bbb7ebec3f1e100b1059b1c6d54bd00e3c1",
    ),
    ExpectedFile(
        "processor_config.json", 1191, "33818c7f9e991ad735fd240209f4fa73e6c28c50"
    ),
    ExpectedFile(
        "tokenizer.json",
        19989325,
        "06b9509352d2af50381ab2247e083b80d32d5c0aba91c272ca9ff729b6a0e523",
    ),
    ExpectedFile(
        "tokenizer_config.json", 1075, "0f19934ef65115549ff9629ff2c83d15e40e32d8"
    ),
)
