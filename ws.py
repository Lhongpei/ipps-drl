from utils.sol_convert import drl_to_ws
import os
for problem in os.listdir("data_test/kim/problem"):
    drl_to_ws("data_test/kim", problem)