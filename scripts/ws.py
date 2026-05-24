from ipps_drl.utils.sol_convert import drl_to_ws
import os
for problem in os.listdir("data/test/kim/problem"):
    drl_to_ws("data/test/kim", problem)