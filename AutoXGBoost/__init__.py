__all__ = ("AutoXGBoost", "AutoXGBoostImputer", "Model", "formats", "models")

from collections import OrderedDict
from pathlib import Path
import os
from Chassis import Chassis, MissingColumnsExplainer, StrOrStrIter, allowAcceptMultipleColumns
from .imports import *
from .defaults import *
from .core import OFDT, Model, hyperparamsTypeCoerce, NotAModelException, UnsupportedFormatException, formatFromFileName, MultiColumnModelType
from .models import XGBoostModel
from . import formats
from .crossvalidation import getLastMetricFromMetricsDict, XGBoostNativeCV
from .utils import parseDict, serializeDict
import warnings

gridSpecModule = lazyImport(".gridSpec")


from lazily import UniOpt
from lazily.sklearn.model_selection import train_test_split
import pathvalidate
from copy import deepcopy

#core.formatExtMapping:Mapping[str, OFDT]={f.ext:f for f in formats if f.ext}
#core.ctorFormatMapping:Mapping[Type[Model], OFDT]={f.ctor:f for f in formats if f.ext}
glue.formatExtMapping.update({f.ext: f for f in formats if f.ext})
glue.ctorFormatMapping.update({f.ctor: f for f in formats if f.ext})


class AutoXGBoost(Chassis):
	"""The class doing most of preparing data to regresion."""

	__slots__ = ("bestHyperparams", "models", "scores", "stop", "modelsPrefix", "hyperparamsFile", "params", "details")
	#bestHyperparams:Mapping[str, Mapping[str, object]]
	#models:Mapping[str, Model]
	#scores:Mapping[str, Tuple[float, float]]
	#modelsPrefix:Path
	#hyperparamsFile:Path
	#params:Mapping[str, typing.Any]?
	#details:typing.Any?

	groupsTypes = type(Chassis.groupsTypes)(Chassis.groupsTypes)
	groupsTypes.update(defaults.categoriesSubtypes)

	def __init__(self, spec: Mapping[str, str], dataset: pandas.DataFrame, params: typing.Optional[Mapping[str, object]] = None, prefix: typing.Optional[Path] = None) -> None:
		"""
		`params` are default params, including hyperparams,
		`prefix` is the path to save and load model files from/to
		"""
		if params is None:
			params = type(defaultParams)(defaultParams)

		super().__init__(spec, dataset)
		self.params = params
		if isinstance(dataset, __class__):
			self.bestHyperparams = deepcopy(dataset.bestHyperparams)
			self.models = deepcopy(dataset.models)
			self.scores = type(dataset.scores)(dataset.scores)
		else:
			self.bestHyperparams = {}
			self.models = {}
			self.scores = OrderedDict()
		
		if prefix is None:
			if isinstance(dataset, __class__):
				prefix = dataset.modelsPrefix
			else:
				prefix = defaultModelsPrefix
		self.modelsPrefix = Path(prefix)
		self.hyperparamsFile = self.modelsPrefix / defaultHyperparamsFileName

	def getColumnsWithMissingHyperparams(self, columns=None) -> Set[str]:
		if columns is None:
			columns = self.columns
		else:
			columns = set(columns)
		return columns - set(self.bestHyperparams.keys())

	def getIssues(self) -> Dict[str, Union[bool, Set[str]]]:
		"""Diagnoses problems in the state of the tool"""
		stopColumnsProcessed = {}
		for pn in ("bestHyperparams", "models", "scores"):
			scIn = self.groups["stop"] & set(getattr(self, pn).keys())
			if scIn:
				stopColumnsProcessed[pn] = scIn

		return {
			"No data": self.pds is None or not bool(len(self.pds)),
			"stop columns processed": stopColumnsProcessed,
			"HP unknown": self.getColumnsWithMissingHyperparams(),
			"HP known, but no models": set(self.bestHyperparams.keys())-set(self.models.keys()),
			"not scored": set(self.models.keys())-set(self.scores.keys()),
		}

	def _reprContents(self) -> str:
		prRes = []
		for prN, prC in self.getIssues().items():
			if prC:
				if not isinstance(prC, bool):
					prRes.append(prN + ": " + str(prC))
				else:
					prRes.append(prN)

		return super()._reprContents() + ((" ! " + ", ".join(prRes)) if prRes else " | OK")

	def getMetric(self, cn: str) -> str:
		"""Returns loss metric suitable for the data type"""
		return metricsMapping[self.features[cn]]

	def prepareCrossValidation(self, cn: str, cns: typing.Optional[typing.Iterable[str]]=None, hyperparams: typing.Optional[dict] = None, metric: typing.Optional[str] = None, testShare: float = 0.0, excludeColumns:typing.Set[str] = None, multiColumnType:MultiColumnModelType=MultiColumnModelType.imput) -> XGBoostNativeCV:
		if metric is None:
			metric = self.getMetric(cn)
		return XGBoostNativeCV(self, cn, cns, hyperparams=hyperparams, metrics=(metric,), testShare=testShare, excludeColumns=excludeColumns, multiColumnType=multiColumnType)

	def getErr(self, cn: str, folds: int = 10, hyperparams: typing.Optional[dict] = None, metric: typing.Optional[str] = None) -> Tuple[np.float64, np.float64]:
		return getLastMetricFromMetricsDict(self.prepareCrossValidation(cn, hyperparams=hyperparams, metric=metric).evaluate(folds=folds, hyperparams=hyperparams))


	def _prepareFittingOneColumn(self, X: pandas.DataFrame, y: pandas.DataFrame, cn: str, testShare: float = 0, strata: typing.Optional[str] = None, weights: typing.Optional[str] = None):
		paramz = type(self.params)(depParams[self.features[cn]])
		paramz.update(self.params)

		if cn in self.catRemap:
			#paramz['num_class'] = len(y.value_counts().index)
			paramz["num_class"] = len(self.catRemap[cn])
			y = y.cat.codes
		#rebalancer=SMOTE()
		#rebalancer._force_all_finite = False
		#(X, y, weights) = rebalancer.fit_sample(X, y, weights) #FUCK, doesn't tolerate missing values

		testSet = None
		if testShare:
			if strata:
				strata = X[strata]
			trainX, testX, trainY, testY, trainWeights, testWeights = train_test_split(X, y, weights, test_size=testShare, stratify=strata)
			if weights is None:
				trainWeights = testWeights = None

			trainSet = (trainX, trainY, trainWeights)
			testSet = (testX, testY, testWeights)
		else:
			trainSet = (X, y, weights)

		return (trainSet, testSet, paramz)
	
	def prepareFitting(self, cns: StrOrStrIter, testShare: float = 0, strata: typing.Optional[str] = None, weights: typing.Optional[str] = None, excludeColumns:typing.Set[str] = None, multiColumnType:MultiColumnModelType=MultiColumnModelType.imput) -> Tuple[pandas.DataFrame, pandas.Series, Dict[str, Union[int, str]]]:
		if isinstance(cns, str):
			cns = (cns, )
		
		if multiColumnType is MultiColumnModelType.imput:
			res = {}
			for cn in cns:
				res.update(self.prepareFitting((cn,), testShare, strata, weights, multiColumnType=MultiColumnModelType.multiOutput))
			return res
		elif multiColumnType is MultiColumnModelType.multiOutput:
			dmat = self.pds.loc[self.colsNotNA(cns)]
			X = self.prepareCovariates(cns, dmat, correctOrder=False, excludeColumns=excludeColumns)
			Y = self.prepareResults(cns, dmat)
			
			if weights is None:
				weights = self.weights
			elif isinstance(weights, str):
				weightsCn = weights
				weights = self.prepareResults(weightsCn, X)
				X = self.prepareCovariates(weightsCn, X, correctOrder=False, excludeColumns=excludeColumns)
			elif isinstance(weights, np.ndarray):
				pass
			else:
				raise TypeError("`weights` is of invalid type `" + weights.__class__.__name__ + "`")
			
			if isinstance(weights, pandas.DataFrame):
				assert len(weights.columns) == 1
				weights = weights[weights.columns[0]]
			
			return {cn:self._prepareFittingOneColumn(X=X, y=Y[cn], cn=cn, testShare=testShare, strata=strata, weights=weights) for cn in cns}
		else:
			raise ValueError(multiColumnType)

	def getSklearnModelClass(self, cn: str):
		"""Returns xgboost.sklearn class suitable for this variable type"""
		if cn in self.groups["categorical"]:
			return xgb.sklearn.XGBClassifier
		else:
			return xgb.sklearn.XGBRegressor

	def bestNumberOfTreesFromCV(self, cn, hyperparams, folds, maxBoostRounds, cv=None):
		if cv is None:
			cv = self.prepareCrossValidation(cn)
		hpNew = type(hyperparams)(hyperparams)
		hpNew["num_boost_round"] = maxBoostRounds
		fromNumTrees = cv.evaluate(folds=folds, hyperparams=hpNew)
		for mName in fromNumTrees:
			return np.argmin(fromNumTrees[mName]["test"]["mean"]) + 1

	def getBestHyperparamsForTheFeature(self, cn: str, cns: typing.Optional[typing.Iterable[str]]=None, folds: int = 10, iters: int = 1000, jobs: int = 3, optimizer: Type["UniOpt.core.Optimizer"] = None, gridSpec: typing.Optional[Mapping[str, object]] = None, testShare: float = 0.0, pointsStorage: "UniOpt.corePointsStorage" = None, stagesMaxRounds=(10, 10000), excludeColumns:typing.Set[str] = None, multiColumnType:MultiColumnModelType=MultiColumnModelType.imput, additionalArgsToCV=None, *args, **kwargs):
		"""A dispatcher method optimizing hyperparams for the column. method is a string used to choose a method"""
		if gridSpec is None:
			gridSpec = dict(gridSpecModule.defaultGridSpec)

		if optimizer is None:
			optimizer = UniOpt.MSRSM
		
		if pointsStorage is None:
			from UniOpt.core.PointsStorage import SQLiteStorage
			uniOptPrefix = self.modelsPrefix / "UniOpt"
			uniOptPrefix.mkdir(mode=0o600, parents=True, exist_ok=True)
			storageFileName = uniOptPrefix / (pathvalidate.sanitize_filename(cn, "_")+".sqlite")
			pointsStorage = SQLiteStorage(storageFileName)

		gridSpec["num_boost_round"] = stagesMaxRounds[0]

		cv = self.prepareCrossValidation(cn, cns, testShare=testShare, excludeColumns=excludeColumns, multiColumnType=multiColumnType, hyperparams=additionalArgsToCV)

		def doCv(hyperparams):
			res = cv.evaluate(folds=folds, hyperparams=hyperparams)
			res = getLastMetricFromMetricsDict(res)
			return res
		
		if iters:
			opt = optimizer(doCv, gridSpec, iters=iters, jobs=jobs, pointsStorage=pointsStorage, *args, **kwargs)
			# TODO: use the following line for hyperband
			#hyperparams["max_depth"]+=round(math.log2(n_iterations))

			best = opt()
			self.details = opt.details
		else:
			best = self.bestHyperparams[cn]
		
		best["num_boost_round"] = self.bestNumberOfTreesFromCV(cn, best, folds, stagesMaxRounds[1], cv=cv)

		#print("best: ", best)
		#print("details: ", self.details)

		hyperparamsTypeCoerce(best)
		assert best is not None
		return best

	def loadHyperparams(self, fileName: typing.Optional[Path] = None) -> None:
		if fileName is None:
			fileName = self.hyperparamsFile
		self.bestHyperparams = parseDict(fileName)

	def saveHyperparams(self, fileName: typing.Optional[Path] = None, append:bool = True) -> None:
		if fileName is None:
			fileName = self.hyperparamsFile
		dictToSave = None
		if append and fileName.is_file():
			def getStuff():
				nonlocal dictToSave
				dictToSave = parseDict(fileName)
				dictToSave.update(self.bestHyperparams)
			#try:
				# this shit just doesn't work right
				#from filelock import FileLock
				#with FileLock(fileName):
					#getStuff()
					#serializeDict(fileName, dictToSave)
					#return
			#except ImportError:
			getStuff()
		else:
			dictToSave = self.bestHyperparams
		serializeDict(fileName, dictToSave)

	def optimizeHyperparams(self, autoSave: bool = True, folds: int = 10, iters: int = 1000, jobs: int = None, optimizer: Type["UniOpt.core.Optimizer"] = None, columns: typing.Optional[Set[str]] = None, force: Optional[bool] = None, testShare: float = 0.0, pointsStorage: "UniOpt.corePointsStorage" = None, additionalArgsToCV=None, *args, **kwargs):
		"""Optimizes hyperparams.
		AutoSave saves best hyperparms after they were estimated for each feature. It's a safety measure for the case of crashes.
		"""
		if columns is None:
			columns = self.columns
		columnsWithPresentHyperparams = columns & set(self.bestHyperparams)
		columnsWithMissingHyperparams = self.getColumnsWithMissingHyperparams()

		columnsToOptimizeHyperparamsFor = columnsWithMissingHyperparams & columns
		if columnsWithPresentHyperparams:
			if force is None:
				warnings.warn("Set `force` into True or False to explicitly clarify if you wanna recompute params which are already present.")
			if force:
				columnsToOptimizeHyperparamsFor |= columnsWithPresentHyperparams
		
		if not iters: # only optimize boosters rounds count
			if not self.bestHyperparams:
				warnings.warn("Best hyperparams must be known if you wanna tune only boosters rounds count. Loading them.")
				self.loadHyperparams()
			missingColumns = self.getColumnsWithMissingHyperparams(columnsToOptimizeHyperparamsFor)
			if missingColumns:
				raise ValueError("Tuning only boosters, but hyperparams are missing for the following columns: "+repr(missingColumns))

		with mtqdm(columnsToOptimizeHyperparamsFor, desc="optimizing hyperparams", unit="model") as pb:
			for cn in pb:
				print("cn:",cn, file=pb)
				#try:
				self.bestHyperparams[cn] = self.getBestHyperparamsForTheFeature(cn, folds=folds, iters=iters, jobs=jobs, optimizer=optimizer, testShare=testShare, pointsStorage=pointsStorage, additionalArgsToCV=additionalArgsToCV, *args, **kwargs)
				if autoSave:
					self.saveHyperparams()
				#except Exception as e:
				#	print(cn, e, file=pb)
				#	#raise e

	def getModelDefaultFileName(self, cn: str, format: OFDT, prefix: Path = None) -> Path:
		if prefix is None:
			prefix = self.modelsPrefix
		return prefix / (pathvalidate.sanitize_filename(cn, "_") + "." + format.ext)

	def loadModel(self, fn: Path = None, cn: str = None, prefix: typing.Optional[Path] = None, format: typing.Optional[OFDT] = None, modelCtor: typing.Type[Model] = None) -> None:
		"""Loads model into an object. Each model is an xgboost binary file accompanied with a json file containing the metadata:
				columns names, their order, their types, and some info about the model.
				if you wanna XGBoost sklearn-like interface, set modelCtor to SKLearnXGBoostModel
		"""
		if isinstance(format, str):
			format = getattr(formats, format)

		if fn is not None:
			if format is None:
				format = formatFromFileName(fn)

		if fn is None:
			if format:
				fn = self.getModelDefaultFileName(cn, format, prefix)
			else:
				if prefix is None:
					prefix = self.modelsPrefix
				for fnCandidate in prefix.glob(pathvalidate.sanitize_filename(cn, "_") + ".*"):
					if format is None:
						format = formatFromFileName(fnCandidate)
						fn = fnCandidate
						if format is not None:
							break

		if modelCtor is None:
			if format and format.ctor:
				modelCtor = format.ctor
			else:
				raise UnsupportedFormatException("load", format, format.ctor)

		self.loadModel_(fn=fn, cn=cn, prefix=prefix, format=format, modelCtor=modelCtor)

	def loadModel_(self, fn: Path = None, cn: str = None, prefix: typing.Optional[Path] = None, format: typing.Optional[OFDT] = None, modelCtor: typing.Type[Model] = None) -> None:
		m = modelCtor(self)
		m.open(fn, cn, prefix, format)

	def convertModel(self, cn: typing.Optional[str], modelCtor: typing.Type[Model]) -> None:
		self.models[cn] = self.models[cn].convert(modelCtor)

	def convertModels(self, modelCtor: typing.Type[Model]) -> None:
		for cn in self.models:
			convertModel(cn, modelCtor)

	def loadModels_(self, prefix: typing.Optional[Path], format: OFDT = None, modelCtor: typing.Type[Model] = None):
		"""Loads all models of specified format from the dir"""
		modelsFileNames = [fn for fn in prefix.glob("*." + format.ext)]
		for fn in modelsFileNames:
			try:
				self.loadModel(fn=fn, prefix=prefix, format=format, modelCtor=modelCtor)
			except NotAModelException:
				pass

	def loadModels(self, prefix: typing.Optional[Path] = None, format: typing.Optional[OFDT] = None, modelCtor: typing.Type[Model] = None):
		if prefix is None:
			prefix = self.modelsPrefix
		if format is None:
			for format in formats:
				self.loadModels_(prefix, format, modelCtor)
		else:
			self.loadModels_(prefix, format, modelCtor)

		self.sortScores()

	def saveModels(self, prefix: typing.Optional[Path] = None, format: typing.Optional[OFDT] = None):
		"""Saves models for all the columns. format allos to serialize models into other formats than binary."""
		if prefix is None:
			prefix = self.modelsPrefix

		with mtqdm(self.models.items(), unit="model", desc="saving models") as pb:
			for cn, m in pb:
				print(cn, file=pb)
				m.save(cn, fn=None, prefix=prefix, format=format)

	def setupEarlyStoppingAndTestSet(self, testSet, pythonHyperparams, total=None):
		if testSet:
			testMat = xgb.DMatrix(testSet[0], label=testSet[1], weight=testSet[2], nthread=-1)
			if "evals" not in pythonHyperparams:
				pythonHyperparams["evals"] = []

			pythonHyperparams["evals"].append((testMat, "earlyStop"))

		pythonHyperparams["early_stopping_rounds"] = 10

	def trainModels(self, cns: typing.Optional[StrOrStrIter] = None, testShare=0., strata=None, weights=None, multiColumnType:MultiColumnModelType=MultiColumnModelType.imput, excludeColumns:typing.Set[str] = None, **kwargs) -> typing.Tuple[XGBoostModel, ...]:
		"""Trains an XGBoost models to predict columns based on the rest of columns (se comments for `MultiColumnModelType`)"""
		if cns is None:
			cns = set(self.bestHyperparams) & set(self.columns)
		if isinstance(cns, str):
			cns = (cns,)
		
		prepared = self.prepareFitting(cns, testShare=testShare, strata=strata, weights=weights, multiColumnType=multiColumnType, excludeColumns=excludeColumns)
		
		ress = {}
		with mtqdm(prepared.items(), desc="training models", unit="model") as columnsPb:
			for cn, preparation in columnsPb:
				print(cn, file=columnsPb)
				((x, y, weights), testSet, paramz) = preparation

				if cn in self.bestHyperparams:
					paramz.update(self.bestHyperparams[cn])

				hyperparamsTypeCoerce(paramz)

				modelClass = XGBoostModel
				pythonHyperparams = modelClass.remapPythonHyperparams(paramz)
				pythonHyperparams.update(kwargs)
				
				xm = xgb.DMatrix(x, label=y, weight=weights, nthread=-1)

				if "callbacks" not in pythonHyperparams:
					pythonHyperparams["callbacks"] = []

				if "verbose_eval" not in pythonHyperparams:
					pythonHyperparams["verbose_eval"] = True

				total = None
				for roundsParamName in ("num_boost_round", "n_estimators"):
					if roundsParamName in pythonHyperparams:
						total = pythonHyperparams[roundsParamName]
						break

				if total is None:
					total = pythonHyperparams["num_boost_round"] = 50

				if testSet:
					self.setupEarlyStoppingAndTestSet(testSet, pythonHyperparams, total)

				if pythonHyperparams["verbose_eval"] is True:
					pythonHyperparams["verbose_eval"] = total // 10
				if isinstance(pythonHyperparams["verbose_eval"], int):
					printInfoEachNIters = pythonHyperparams["verbose_eval"]
				else:
					printInfoEachNIters = 0
				pythonHyperparams["verbose_eval"] = False

				with mtqdm(total=total, desc="fitting model for col " + cn, unit="round") as modelPb:
					showedI = -1

					def progressCallback(env):
						nonlocal showedI
						i = env.iteration
						delta = i - showedI
						showedI = i

						if env.evaluation_result_list and printInfoEachNIters and i % printInfoEachNIters == 0:
							print(dict(env.evaluation_result_list), file=modelPb)
						modelPb.update(delta)

					pythonHyperparams["callbacks"].append(progressCallback)
					m = xgb.train(paramz, xm, **pythonHyperparams)

				res = modelClass(self, m)
				res.columnName = cn
				self.models[cn] = res
				ress[cn] = res
		return ress
	

	def sortScores(self) -> None:
		"""Sorts scores from the best to the worst"""
		self.scores = type(self.scores)(sorted(self.scores.items(), key=lambda x: abs(x[1][0]) * 100 + abs(x[1][1])))

	def scoreModels(self, folds: int = 20, columns: typing.Optional[StrOrStrIter] = None) -> None:
		"""Computes errors for each model using n-fold crossvalidation."""
		if columns is None:
			columns = self.models
		if isinstance(columns, str):
			columns = (columns, )
		with mtqdm(columns, desc="scoring models") as pb:
			for cn in pb:
				print(cn, file=pb)
				self.scores[cn] = self.getErr(cn, folds=folds)
				scale = typeScoreCoefficients[self.features[cn]](self, cn) ** 2
				if scale:
					self.scores[cn] = (self.scores[cn][0] / scale, self.scores[cn][1] / scale)
		self.sortScores()
		self.scores = OrderedDict(self.scores)

	
	def prepareCovariates(self, cns: str, dmat: typing.Optional[pandas.DataFrame] = None, correctOrder: bool = True, excludeColumns:typing.Set[str] = None) -> pandas.DataFrame:
		if isinstance(cns, str):
			cns = (cns,)
		x = super().prepareCovariates(cns, dmat, excludeColumns=excludeColumns)
		
		if correctOrder:
			cns = set(cns)
			rightFeatureNamesOrder = None
			for cn in set(self.models) & cns:
				m = cast(XGBoostModel, self.models[cn])
				if rightFeatureNamesOrder is not None:
					if rightFeatureNamesOrder != m.featureNames:
						raise ValueError("Models have features in different order. Since we are fitting multiple-output regression/classification, the matrix must be exactly in the same order. (In fact we can reorder for each fitting individually, but it is pointless - when we fit a multiple-output regression all the stuff will be in correct order. When a user tampers, it's his responsibility to make the stuff have the correct order)")
				else:
					rightFeatureNamesOrder = m.featureNames
			
			if rightFeatureNamesOrder is not None:
				missingColumnsInDmat = set(rightFeatureNamesOrder) - set(x.columns)
				mce = MissingColumnsExplainer(self, missingColumnsInDmat)
				mce.fix(x)
			else:
				# get the order from any model for compatibility
				for m in self.models.values():
					rightFeatureNamesOrder = [cn for cn in m.featureNames if cn not in cns]
					break
			
			if rightFeatureNamesOrder is not None:
				try:
					with warnings.catch_warnings():
						warnings.simplefilter("ignore")
						x = x.loc[:, rightFeatureNamesOrder]
				except:
					x = x.reindex(rightFeatureNamesOrder)
		return x

	def preparePrediction(self, cn: str, dmat: typing.Optional[Chassis] = None, excludeColumns:typing.Set[str] = None, **kwargs):
		if dmat is None:
			dmat = self
		obj = None
		if isinstance(dmat, pandas.DataFrame):
			dmat = dmat
			obj = self
		elif isinstance(dmat, Chassis):
			obj = dmat
			dmat = dmat.pds
		else:
			raise Exception("dmat must be either a pandas.DataFrame or a Chassis")

		x = self.prepareCovariates(cn, dmat, correctOrder=True, excludeColumns=excludeColumns)
		m = cast(XGBoostModel, self.models[cn])
		return x, m
	
	def predictSingleColumn(self, cn: str, dmat: typing.Optional[Chassis] = None, argmax: bool = True, returnPandas: bool = None, SHAPInteractions: typing.Optional[bool] = None, **kwargs):
		"""Predicts column values based on the matrix dmat. argmax controls whether we should apply argmax for one-hot encoded categorical variables rather than return a probability vector."""
		x, m = self.preparePrediction(cn, dmat, **kwargs)
		res = m.predict(x, **kwargs)

		if SHAPInteractions is not None:
			shapRes = self.computeSHAPValues(cn=cn, dmat=None, argmax=argmax, returnPandas=returnPandas, SHAPInteractions=SHAPInteractions, _xm=(x, m), **kwargs)

		if not isinstance(x.index, pandas.core.indexes.range.RangeIndex) or x.index._start != 0:
			if returnPandas is False:
				warnings.warn("Source dataset uses discontinued index. Make sure that the source array is sorted by index (sort_index in pandas).")
			else:
				returnPandas = True

		#print(res)
		if cn in self.groups["categorical"] and argmax:
			res = self.oneHotToCategory(cn, res, x.index)
			if returnPandas is False:
				res = res.as_matrix()
		else:
			if returnPandas is True:
				res = self.numpyToColumn(cn, res, x.index)

		if SHAPInteractions is None:
			return res
		else:
			return res, shapRes

	def predict(self, cns: StrOrStrIter, dmat: typing.Optional[Chassis] = None, argmax: bool = True, returnPandas: bool = None, SHAPInteractions: typing.Optional[bool] = None, **kwargs) -> np.ndarray:
		"""Predicts columns values based on the matrix dmat. argmax controls whether we should apply argmax for one-hot encoded categorical variables rather than return a probability vector."""
		if not isinstance(cns, str):
			assert returnPandas is not None, "Different columns may have automatically decided differently about whether to return a `pandas.DataFrame` or numpy array. Make this decision yourself!"
			if returnPandas:
				shapExpls = {}
				res = []
				for cn in cns:
					cRes = self.predictSingleColumn(cn, dmat, argmax, returnPandas, SHAPInteractions, **kwargs)
					
					if SHAPInteractions is not None:
						cRes, shapExpls[cn] = cRes
					
					res.append(cRes)
				res = pandas.concat(res, axis=1)
				
				if SHAPInteractions is not None:
					return res, shapExpls
				else:
					return res
			else:
				return {cn: self.predict(cn, dmat, argmax, returnPandas, SHAPInteractions, **kwargs) for cn in cns}
		else:
			cn = cns
			return self.predictSingleColumn(cn, dmat, argmax, returnPandas, SHAPInteractions, **kwargs)

	def computeSHAPValues(self, cn: str, dmat: typing.Optional[Chassis] = None, argmax: bool = True, returnPandas: bool = None, SHAPInteractions: typing.Optional[bool] = False, _xm=None, **kwargs):
		"""Computes shap values and interactions"""
		if _xm is None:
			_xm = self.preparePrediction(cn, dmat, **kwargs)
		x, m = _xm

		if SHAPInteractions is False:
			shapValues = m.predict(x, pred_contribs=True, **kwargs)
		elif SHAPInteractions is True:
			shapValues = m.predict(x, pred_interactions=True, **kwargs)

		if not isinstance(x.index, pandas.core.indexes.range.RangeIndex) or x.index._start != 0:
			if returnPandas is False:
				warnings.warn("Source dataset uses discontinued index. Make sure that the source array is sorted by index (sort_index in pandas).")
			else:
				returnPandas = True

		if not SHAPInteractions:
			biasCol = shapValues[:, -1]
			shapValues = shapValues[:, :-1]
			if returnPandas is True:
				shapValues = pandas.DataFrame(shapValues, columns=m.featureNames)
		else:
			biasCol = shapValues[:, -1]
			biasRow = shapValues[-1, :]
			shapValues = shapValues[:-1, :-1]
			if returnPandas is True:
				warnings.warn("SHAP interactions is 3d stuff unsuitable for pandas")

		#print(res)
		if argmax:
			warnings.warn("Not sure that I have implemented combining categorical SHAP values correctly. There may be problems")
			shapValues = self.reduceCategoricalCols(shapValues)
			if returnPandas is False:
				shapValues = shapValues.as_matrix()

		if not SHAPInteractions:
			return (shapValues, biasCol)
		else:
			return (shapValues, biasCol, biasRow)

	def SHAPForcePlot(self, shapValuesAndBiases=None, cn: str = None, dmat: typing.Optional[Chassis] = None, argmax: bool = True, _xm=None, **kwargs):
		import shap

		if shapValuesAndBiases is None:
			shapValuesAndBiases = self.computeSHAPValues(self, cn=cn, dmat=dmat, argmax=argmax, returnPandas=True, SHAPInteractions=False, **kwargs)
		shapValues, biases = shapValuesAndBiases
		return shap.force_plot(biases[0], shapValues.values, feature_names=shapValues.columns)


class AutoXGBoostImputer(AutoXGBoost):
	"""This is an imputer. It imputs missing values into dataset based on XGBoost-trained model"""

	def fillMissing(self, cn: str, dmat=None):
		"""Finds values missing in the column, predicts them and imputs."""
		if dmat is None:
			dmat = self
		sel = dmat.colsIsNA(cn)
		v = dmat.pds.loc[sel]
		if len(v):
			res = self.predict(cn, dmat=v, argmax=False)
			if cn in self.groups["categorical"]:
				cn = self.catRemap[cn]
			dmat.pds.loc[sel, cn] = res

	def imput(self, columns: StrOrStrIter = None, dmat=None):
		"""Imputs values into all the columns iteratively starting from the best-performing models."""
		if dmat is None:
			dmat = self
		if columns is None:
			if not self.scores:
				raise Exception("`scoreModels` first!!!")
			columns = set(sorted(set(self.models), key=lambda k: self.scores[k], reverse=True))
			
		if isinstance(columns, str):
			columns = (columns, )
		
		stoppedColumns = set(columns) & self.groups["stop"]
		if stoppedColumns:
			warnings.warn("Some columns to be processed are in `stop` category. " + repr(stoppedColumns) + " Ignoring them.")
			columns -= stoppedColumns

		with mtqdm(columns, desc="imputting") as pb:
			for cn in pb:
				self.fillMissing(cn, dmat=dmat)