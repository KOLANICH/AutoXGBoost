__all__ = ("getLastMetricFromMetricsDict", "XGBoostNativeCV")


import typing
from collections import defaultdict
from .imports import *
from .core import hyperparamsTypeCoerce, MultiColumnModelType
from .models import XGBoostModel


def decodeMetricsDF(errorDf: pandas.DataFrame) -> Mapping[str, Mapping[str, pandas.DataFrame]]:
	res = defaultdict(lambda: defaultdict(dict))
	for cn in errorDf.columns:
		parts = cn.split("-")
		testTr = parts[0]
		moment = parts[-1]
		mName = "-".join(parts[1:-1])
		res[mName][testTr][moment] = errorDf.loc[:, cn]
	return res


def getLastMetricFromMetricsDict(decodedMetricsDict: defaultdict, part: str = "test") -> Tuple[np.float64, np.float64]:
	for mName in decodedMetricsDict:
		df = decodedMetricsDict[mName][part]
		l = len(df["mean"])
		if l:
			return (df["mean"][l - 1], df["std"][l - 1])
		else:
			return (np.inf, np.inf)


class XGBoostNativeCV:
	def __init__(self, parent: "AutoXGBoost", cn: str, cns: typing.Optional[typing.Iterable[str]]=None, hyperparams: typing.Optional[dict] = None, metrics: typing.Tuple[str] = ("error",), testShare: float = 0.0, weights=None, excludeColumns:typing.Set[str] = None, multiColumnType:MultiColumnModelType=MultiColumnModelType.imput) -> None:
		if cns is None:
			cns = (cn,)
		
		preparationsDict = parent.prepareFitting(cns, testShare=testShare, weights=weights, excludeColumns=excludeColumns, multiColumnType=multiColumnType)
		((x, y, weights), testSet, self.hyperparams) = preparationsDict[cn]
		#print("\x1b[31m  weights", weights, "\x1b[0m")
		
		if not len(y):
			raise Exception("Nothing to fit! " + cn + " is EMPTY!")
		self.rowsInDMat = len(y)
		self.cn = cn
		self.xm = xgb.DMatrix(x, label=y, weight=weights)

		if hyperparams is None:
			hyperparams = {}

		parent.setupEarlyStoppingAndTestSet(testSet, hyperparams)

		if hyperparams:
			self.hyperparams.update(hyperparams)

		self.metrics = metrics
		self.parent = parent

	def evaluate(self, folds: int = 10, hyperparams: typing.Optional[dict] = None) -> typing.Mapping[str, float]:
		if folds > self.rowsInDMat:
			#warn(RuntimeWarning("count of folds ("+str(folds)+") MUST be less than count of rows in mat ("+str(self.rowsInDMat)+"), reducing folds to "+str(self.rowsInDMat)+"..."))
			folds = self.rowsInDMat

		if not hyperparams:
			hyperparams = self.parent.bestHyperparams[self.cn]
		paramz = type(self.hyperparams)(self.hyperparams)
		paramz.update(hyperparams)
		paramz["eval_metric"] = self.metrics[0]

		hyperparamsTypeCoerce(paramz)
		#pprint(paramz)

		if self.cn in self.parent.models:
			modelClass = self.parent.models[self.cn].__class__
		else:
			modelClass = XGBoostModel

		pythonHyperparams = modelClass.remapPythonHyperparams(paramz)
		#print("\n\pythonHyperparams:", pythonHyperparams)
		#print("\n\nparamz:", paramz)

		cvRes = xgb.cv(paramz, self.xm, nfold=folds, metrics=self.metrics, **pythonHyperparams)
		res = decodeMetricsDF(cvRes)
		return res