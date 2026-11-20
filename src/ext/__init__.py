"""Phase 3 extensions (team decisions beyond the Assignment 3 spec).

E1  heavier noise: rates up to 0.90 with the same a2_default rule, seeds and RNG key   (conditions.py)
E2  true loss S / T_clean, n-gram control and geometry on every condition             (scoring.py, geometry.py)
E3  hidden-layer x epoch sweep of the two FFNNs, width fixed at 64                      (sweep.py)
E4  trace any sentence through the whole pipeline ("A cow jumps over the fence")        (trace.py)

Nothing here modifies an Assignment 2 or Assignment 3 artifact. Run everything with
`python phase3_extensions.py`; results land in results/phase3/ext/.
"""
