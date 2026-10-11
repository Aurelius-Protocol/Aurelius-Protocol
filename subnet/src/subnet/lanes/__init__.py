# Miner dependency subset extracted from Aurelius v5 (02375403).
"""subnet.lanes — task-type adapters (LANE code).

Everything under here is lane-specific and MUST NOT be imported by any platform module
(contract `platform-not-lane`). Adding a new lane here must require ZERO platform changes — that
is the acceptance test for the platform/lane boundary (plan §5 / risk #22).
"""
