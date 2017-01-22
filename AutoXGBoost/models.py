from pathlib import Path
import os
from .imports import *
from .core import Model, OFDT, NotAModelException
#from .core import *

from .utils import IterableModule
from . import formats
from .utils import parseDict, serializeDict

pyxgboostCompiler = lazyImport("pyxgboost.compiler")

modelSerializationSaveRestoreFields = {"feature_names", "feature_types"}


def xgboostImportBinary(self, bts: bytes) -> xgb.Booster:
	bts = bytearray(bts)
	m = xgb.Booster(params=self.params, model_file=bts)
	m = xgb.Booster(params=self.params, model_file=bts)  # after the first load it predicts incorrectly, seems to be a bug in XGBoost
	return m


def pythonImportPython(source: bytes):
	source = source.decode("utf-8")
	assert source, "Source is EMPTY! WTF?"
	compiled = compile(source, filename="xgboost model", mode="exec")
	resEnv = {"__builtins__": {"list": list, "dict": dict, "float": float, "int": int, "str": str, "range": range, "enumerate": enumerate, "len": len, "AssertionError": AssertionError}}
	exec(compiled, resEnv, resEnv)
	return resEnv["predict"]


class Models(IterableModule):
	__all__ = ("XGBoostModel", "SKLearnXGBoostModel", "pyXGBoostModel")

	class XGBoostModelMetadata(Model):
		#cn:str #column name
		def __init__(self, parent, model: xgb.Booster = None):
			super().__init__(parent, model)

		@staticmethod
		def getMetaFn(prefix: Path, fn: Path) -> Path:
			return prefix / (fn.stem + ".json")

		@property
		def params(self):
			return self.parent.bestHyperparams[self.columnName]

		@params.setter
		def params(self, val):
			self.parent.bestHyperparams[self.columnName] = val

		def doOpen(self, fn: Path, cn: str, prefix: Optional[Path] = None, format: Optional[OFDT] = None) -> Dict[str, Any]:
			"""Loads model into an object. Each model is an xgboost binary file accompanied with a json file containing the metadata:
					columns names, their order, their types, and some info about the model.
					if sklearn is true, xgboost sklearn-like interface is used.
			"""
			if prefix is None:
				prefix = self.parent.modelsPrefix

			params = type(self.parent.params)(self.parent.params)
			if cn in self.parent.bestHyperparams:
				params.update(self.parent.bestHyperparams[cn])

			metaFn = self.__class__.getMetaFn(prefix, fn)

			attrs = parseDict(metaFn)
			if not attrs:
				raise NotAModelException(fn, cn, format)

			if "params" in attrs:
				params.update(attrs["params"])
			self.parent.bestHyperparams[cn] = params

			if "categories" in attrs:
				self.parent.catIndex[cn] = pandas.Index(attrs["categories"])
				self.parent.catRemap[cn] = [cn + "_" + an for an in attrs["categories"]]

			if "score" in attrs:
				self.parent.scores[cn] = attrs["score"]

			super().doOpen(fn, cn, prefix, format)

			for an in modelSerializationSaveRestoreFields:
				if an in attrs:
					setattr(self.formattedModel, an, attrs[an])
			return attrs

		def doSave(self, cn: str, fn: Path, prefix: Optional[Path] = None, format: Optional[OFDT] = None) -> None:
			"""Saves models with their metadata. See the docstring of loadModel"""
			if prefix is None:
				prefix = self.parent.modelsPrefix
			os.makedirs(str(prefix), exist_ok=True)

			md = {an: getattr(self.formattedModel, an) for an in modelSerializationSaveRestoreFields}
			md["params"] = self.parent.bestHyperparams[cn]

			if cn in self.parent.catIndex:
				md["categories"] = list(self.parent.catIndex[cn])

			if cn in self.parent.scores:
				md["score"] = self.parent.scores[cn]

			super().doSave(cn, fn, prefix, format)

			for an in modelSerializationSaveRestoreFields:
				md[an] = getattr(self.formattedModel, an)

			serializeDict(self.__class__.getMetaFn(prefix, fn), md)

	class XGBoostModel(XGBoostModelMetadata):
		exports = {
			formats._internalXGBoostBoosterFormat: lambda m: m,
			formats.binary: lambda m: m.save_raw(),
			formats.python: lambda m: pyxgboostCompiler.compile(m.save_raw()).encode("utf-8"),
			formats.text: lambda m: "\n".join(m.get_dump(dump_format="text")).encode("utf-8"),
			formats.json: lambda m: "\n".join(m.get_dump(dump_format="json")).encode("utf-8")
		}
		imports = {
			formats._internalXGBoostBoosterFormat: lambda m: m,
			formats.binary: xgboostImportBinary
		}
		nativeRepresentation = xgb.Booster
		hyperparamsNamesRemap = {
			"num_boost_round": "num_boost_round",
			"n_estimators": "num_boost_round",
			"early_stopping_rounds": "early_stopping_rounds",
			"obj": "obj",
			"feval": "feval",
		}

		def __init__(self, *args, **kwargs) -> None:
			super().__init__(*args, **kwargs)
			self.imports[formats.binary] = partial(self.imports[formats.binary], self)

		@property
		def featureNames(self) -> List[str]:
			return self.formattedModel.feature_names

		@featureNames.setter
		def featureNames(self, v):
			self.formattedModel.feature_names = v

		def predict(self, featureVector: Iterable[float], **kwargs) -> np.ndarray:
			x = xgb.DMatrix(featureVector)
			print(kwargs)
			return self.model.predict(x, **kwargs)

	formats.binary.ctor = XGBoostModel

	class SKLearnXGBoostModel(XGBoostModel):
		# hyperparamsNamesRemap=type(XGBoostModel.hyperparamsNamesRemap)(XGBoostModel.hyperparamsNamesRemap)
		hyperparamsNamesRemap = {"early_stopping_rounds": "early_stopping_rounds"}
		hyperparamsNamesRemap.update({
			"num_boost_round": "n_estimators",
			"n_estimators": "n_estimators",
		})

		def doOpen(self, fn: Path, cn: str, prefix: Optional[Path] = None, format: Optional[OFDT] = None) -> None:
			super().doOpen(fn, cn, prefix, format)

		@property
		def formattedModel(self):
			return self.model._Booster

		@formattedModel.setter
		def formattedModel(self, val):
			sklearnModelClass = self.parent.getSklearnModelClass(self.columnName)
			mm = sklearnModelClass(**self.parent.params)
			mm._Booster = val
			self.model = mm
			# _le = ? # from pandas dataframe?
			# we need to save and restore a fitted XGBLabelEncoder somehow
			# https://github.com/dmlc/xgboost/issues/2073

		def predict(self, featureVector: Iterable[float]):
			return self.model.predict(featureVector)

	class pyXGBoostModel(XGBoostModelMetadata):
		imports = {
			formats.python: pythonImportPython
		}
		exports = {}
		nativeRepresentation = types.FunctionType

		def doOpen(self, fn: Path, cn: str, prefix: Optional[Path] = None, format: Optional[OFDT] = None) -> None:
			self.attrs = super().doOpen(fn, cn, prefix, format)

		def __init__(self, parent, model: Callable[[Iterable[float]], float] = None) -> None:
			super().__init__(parent, model)
			self.attrs = None
			self.oFn = None

		@property
		def featureNames(self) -> List[str]:
			return self.attrs["feature_names"]

		@featureNames.setter
		def featureNames(self, v):
			self.attrs["feature_names"] = v

		def predict(self, featureVector: Iterable[float]):
			m = featureVector.values
			return [self.formattedModel(r) for r in m]

	formats.python.ctor = pyXGBoostModel


sys.modules[__name__] = Models(__name__)