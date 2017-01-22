pip3 install --upgrade --user --pre https://gitlab.com/KOLANICH/pyxgboost/-/jobs/artifacts/master/raw/wheels/pyxgboost-0.CI-py3-none-any.whl?job=build
git clone --depth=1 https://gitlab.com/KOLANICH/UniOpt.py.git
source ./UniOpt.py/.ci/installBackendsDependencies.sh
pip3 install --upgrade --user --pre ./[hyperopt,hyperengine,SKOpt,SMAC,BeeColony,optunity,Yabox,PySHAC,RBFOpt,Bayessian]
