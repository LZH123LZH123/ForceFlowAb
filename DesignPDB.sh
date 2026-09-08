
# NO
python -m diffab.tools.runner.design_for_pdb \
  ./testwk/9MBA_U_V.pdb \
  -c ./configs/test/multicdrs.yml \
  --heavy V \
  --light U \
  -o ./testwk \
  -b 32

python -m diffab.tools.runner.design_for_pdb \
  ./testwk/9MBA_U_V.pdb \
  -c ./configs/test/H3.yml \
  --heavy V \
  --light U \
  -o ./testwk \
  -b 32


python -m diffab.tools.runner.design_for_pdb \
  ./testwk/nanobody.pdb \
  --heavy H \
  --light "" \
  -c ./configs/test/nanobody.yml \
  -o ./testwk \
  -b 32


python -m diffab.tools.runner.design_for_pdb \
  ./testwk/nanobody.pdb \
  --heavy H \
  --light "" \
  -c ./configs/test/H3.yml \
  -o ./testwk \
  -b 32


#YES
python -m diffab.tools.runner.design_for_pdb \
  ./testwk/9MBA_U_V.pdb \
  -c ./configs/test/multicdrs_T.yml \
  --heavy V \
  --light U \
  -o ./testwk \
  -b 32

python -m diffab.tools.runner.design_for_pdb \
  ./testwk/9MBA_U_V.pdb \
  -c ./configs/test/H3_T.yml \
  --heavy V \
  --light U \
  -o ./testwk \
  -b 32


python -m diffab.tools.runner.design_for_pdb \
  ./testwk/nanobody.pdb \
  --heavy H \
  --light "" \
  -c ./configs/test/nanobody_T.yml \
  -o ./testwk \
  -b 32


python -m diffab.tools.runner.design_for_pdb \
  ./testwk/nanobody.pdb \
  --heavy H \
  --light "" \
  -c ./configs/test/H3_T.yml \
  -o ./testwk \
  -b 32

# python diffab/tools/relax/run.py --root ./testwk/H3 --pipeline pyrosetta
# python  diffab/tools/eval/run.py  --root  ./testwk/H3 --pfx rosetta

# python diffab/tools/relax/run.py --root ./testwk/multicdrs --pipeline pyrosetta
# python  diffab/tools/eval/run.py  --root  ./testwk/multicdrs --pfx rosetta