"""Reviewed SN37 funding identities shared by terms validation and chain execution.

Code pins are the node's :code storage hashes (BLAKE2b-256), not raw WASM SHA256.
Runtime481 matches the attributed metadata fixture and the deployed deposit verifier.
Unknown runtimes need their own review before they can authorize a purchase.
"""

RUNTIME_PROFILES = {
    475: (
        "e181ddacd13d1050e82a2afea8e58ba5d3fc84504fa92d2884c91df8c9dd8020",
        "0x557634c8c31bc639ea6552e297dcfb781cd7d9bdd491a5352c151305db33d3a0",
    ),
    477: (
        "1d9a056217f440400df497ad70d4a6a6148abc19871d1b28e52464c507c11260",
        "0x18b52484d9c5d7523f704c86b17b2ab959fffbc6edb6766a77e9877f0ad19387",
    ),
    479: (
        "075ee9614fee27fb83a44bda737f6803bd3f6642d57ab492321aa47dad359080",
        "0x44e952fd1cebe9bd15c0c55cee6e1d7c6c97b44d941b79127343118dbbd342bc",
    ),
    481: (
        "4dc0d25bc11d1ea15118b433ba34e78f40ae2bfbaa188757e6f67f0a3240266a",
        "0x5fb504c1c793a3ddb5456f189ddef98b5146853aea6ff998501f1b6c6eb720bc",
    ),
}
