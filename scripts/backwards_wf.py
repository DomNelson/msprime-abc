import sys, os
from itertools import islice
import numpy as np

import wf_sims

B = wf_sims.BackwardSim(10, 1, 0.5, discrete_loci=False)
B.init_pop()

next_recs = B.draw_recomb_vals()
B.P.recombine(next_recs)
B.P.climb(next_recs)
B.P.coalesce()
