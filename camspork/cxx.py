import sys, os, subprocess

env = os.environ
cxx = env.get("EXO_CXX") or env.get("CXX") or "c++"
subprocess.check_call([cxx] + sys.argv[1:])
