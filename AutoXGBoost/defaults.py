from pathlib import Path
from typing import Mapping


depParams = {
	"categorical": {"objective": "multi:softprob"},
	#multi:softmax: set XGBoost to do multiclass classification using the softmax objective, you also need to set num_class(number of classes)
	"numerical": {"objective": "reg:squarederror"},
	"binary": {"objective": "binary:logistic"},
	#reg:logistic https://github.com/dmlc/xgboost/issues/521
	#binary:logitraw
	#binary:hinge
	"survival": {"objective": "survival:cox"},
	"gamma": {"objective": "reg:gamma"},
	"tweedie": {"objective": "reg:tweedie"},
	"poisson": {"objective": "count:poisson"},
	"pairwise": {"objective": "rank:pairwise"},
	"NDCG": {"objective": "rank:ndcg"},
	"MAP": {"objective": "rank:map"},
}


typeScoreCoefficients = {"binary": lambda self, cn: 1, "categorical": lambda self, cn: 1, "numerical": lambda self, cn: self.pds[cn].std()}

metricsMapping = {
	#"categorical": "mlogloss",
	"categorical": "merror",
	#"numerical": "mae",
	"numerical": "rmse",
	#"binary": "auc",
	"binary": "logloss",
	#"binary": "error",
	"survival": "cox-nloglik",
	"poisson": "poisson-nloglik",
	"gamma": "gamma-nloglik",
	#"gamma": "gamma-deviance",
	"tweedie": "tweedie-nloglik",
}

categoriesSubtypes = {
	"survival": "numerical",
	"gamma": "numerical",
	"tweedie": "numerical",
	"poisson": "numerical",
	"pairwise": "numerical"
}


defaultModelsPrefix = Path("./AutoXGBoost_models/")
defaultHyperparamsFileName = "bestHyperparams.json"
#defaultParams:Mapping[str, object]={
defaultParams = {
	"silent": 1
}