"""Reviewed SN37 funding identities shared by terms validation and chain execution.

Code pins are the node's :code storage hashes (BLAKE2b-256), not raw WASM SHA256.
Runtime488 and runtime497 match the independently attributed metadata fixtures and reviewed
funding codecs. Unknown runtimes need their own review before they can authorize a purchase.
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
    488: (
        "66804d7b7f326f3285f900a1533df30c17ffa1d2855a5c3f3559ae05471446b7",
        "0x3147daab804046184359d8a929cb333d75f490323b9e1327887a172c64eafa94",
    ),
    497: (
        "d54cab26911b8c2d840b5bc0534945e43d43cf7be23266d5100f9c0c64ecdd4e",
        "0x1ef826aebed0d77300b7da09d4e3b23ad507317d72c8c01d827537643dbf2c15",
    ),
}

# The transaction version is part of a runtime's code, and the coldkey signs it inside every
# funding transfer. Each reviewed runtime carries the version its capture recorded: 1 through
# runtime488, and 2 from runtime494 on. A chain, a retained intent or a server that names
# another version for one of these runtimes is refused.
RUNTIME_TRANSACTION_VERSIONS = {475: 1, 477: 1, 479: 1, 481: 1, 488: 1, 497: 2}
