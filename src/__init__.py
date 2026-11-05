"""Define the paper task metadata."""

TASK_HEADS = {
    "phrase": frozenset({"phrase"}),
    "structure": frozenset({"section_function", "section_boundary"}),
    "chord": frozenset({"chord_root", "chord_quality"}),
    "beat": frozenset({"tactus", "downbeat"}),
}
INTERNAL_HEADS = frozenset().union(*TASK_HEADS.values())
