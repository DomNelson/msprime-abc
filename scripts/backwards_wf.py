import sys, os
from itertools import islice
import numpy as np

import wf_sims

B = wf_sims.BackwardSim(5, 1, 0.5, discrete_loci=False)
B.init_pop()

max_gens = 100
rec_generator = (B.draw_recomb_vals() for i in range(max_gens))

B.P.trace(rec_generator)
