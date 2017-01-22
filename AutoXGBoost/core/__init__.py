from pathlib import Path
from ..imports import *
from ..glue import formatExtMapping
from enum import IntEnum
from ..glue import formats
from .FormatConverter import *
import pathvalidate

HyperparamsDictT = Mapping[str, Any]


def formatFromFileName(fn):
	ext = fn.suffix[1:]
	if ext in formatExtMapping:
		return formatExtMapping[ext]

class MultiColumnModelType(IntEnum):
	# notation: [all columns in DS] x [selected columns to fit] -> (target column, [feature vector columns]), ...
	imput = 0       # [a, b, c, d] x [a, b] -> (a, [b, c, d]), (b, [a, c, d])
	multiOutput = 1 # [a, b, c, d] x [a, b] -> (a, [c, d]), (b, [c, d])



class Model(object):
	"""A class abstracting a model"""

	__slots__ = ("parent", "model")
	#parent:"AutoXGBoost"
	#model:object #object to be interacted to
	#formattedModel:object #object to be created and dumped by formats
	#exports:Mapping[OFDT, types.FunctionType]=None
	#imports:Mapping[OFDT, types.FunctionType]=None

	@property
	def formattedModel(self):
		return self.model

	@formattedModel.setter
	def formattedModel(self, val):
		self.model = val

	def __init__(self, parent, model: Any = None) -> None:
		super().__init__()
		self.exports = type(self.__class__.exports)(self.__class__.exports)
		self.imports = type(self.__class__.imports)(self.__class__.imports)
		self.parent = parent
		self.columnName = None
		self.model = model

	def preprocessParams(self, format: Optional[OFDT], formatsSet: Set[OFDT], opName: str):
		if isinstance(format, str):
			format = getattr(formats, format)

		if format not in formatsSet:
			raise UnsupportedFormatException(opName, format, self.__class__)
		return format

	def open(self, fn: Path, cn: str, prefix: Optional[Path] = None, format: Optional[OFDT] = None) -> None:
		if fn is None:
			fn = self.parent.getModelDefaultFileName(cn, format, prefix)

		if format is None:
			format = formatFromFileName(fn)

		if cn is None:
			cn = fn.stem

		if prefix is None:
			prefix = fn.parent

		self.doOpen(fn, cn, prefix, format)
		self.parent.models[cn] = self

	def save(self, cn: str = None, fn: Optional[Path] = None, prefix: Optional[Path] = None, format: Optional[OFDT] = None) -> int:
		"""Saves models with their metadata. See the docstring of loadModel"""
		if cn is None:
			cn = self.columnName
		if format is None:
			if fn is not None:
				format = formatFromFileName(fn)
			else:
				format = formats.binary

		if isinstance(format, str):
			format = getattr(formats, format)

		if fn is None:
			fn = self.parent.getModelDefaultFileName(cn, format, prefix)
		else:
			if prefix is None:
				prefix = fn.parent
				if format is None:
					format = formatFromFileName(fn)
			else:
				raise ValueError("When you specify full file name, prefix must be None")

		return self.doSave(cn, fn, prefix, format)

	def predict(self, featureVector: Iterable[float]) -> Iterable[float]:
		"""Transforms input feature vector into output vector"""
		raise NotImplementedError()

	def doOpen(self, fn: Path, cn: str, prefix: Optional[Path] = None, format: Optional[OFDT] = None):
		self.columnName = cn
		with fn.open("rb") as f:
			self.load(format, f.read())

	def doSave(self, cn: str, fn: Path, prefix: Optional[Path] = None, format: typing.Optional[OFDT] = None):
		"""Saves models with their metadata. See the docstring of loadModel"""
		serializedModelBytes = self.dump(format)
		assert serializedModelBytes, "Data dumped is EMPTY! WTF?"
		with fn.open("wb") as f:
			return f.write(serializedModelBytes)

	def load(self, format: OFDT, bts: bytes):
		"""Loads model into an object."""
		assert bts, "bytes of serialized model are EMPTY! WTF?"
		format = self.preprocessParams(format, self.imports, "load")
		self.formattedModel = self.imports[format](bts)

	def dump(self, format: OFDT):
		format = self.preprocessParams(format, self.exports, "dump")
		assert self.formattedModel, "self.formattedModel is None. WTF?"
		bts = self.exports[format](self.formattedModel)
		assert bts, "bytes of serialized model are EMPTY! WTF?"
		return bts

	def convert(self, tp: Type["Model"]) -> "Model":
		transitionFormats = self.exports.keys() & tp.imports.keys()
		#print("convert", self.__class__, "->", tp, transitionFormats)
		if self.__class__ is tp:
			#warnings.warn("conversion from "+self.__class__.__name__+" to itself, returning self")
			return self

		if transitionFormats:
			f = next(iter(transitionFormats))
			#tp.imports[f](self.exports[f](self.formattedModel))
			res = tp(self.parent, None)
			res.columnName = self.columnName
			res.load(f, self.dump(f))
			return res

	@classmethod
	def remapPythonHyperparams(cls, hyperparamsDict: HyperparamsDictT) -> HyperparamsDictT:
		"""XGBoost own fitting interface, and SKLearn-compatible one use different hyperparam names for the hyperparams not passed directly to native code side. No shims are provided by default, so we have to workaround this. This function moves these (listed in `__class__.hyperparamsNamesRemap`) hyperparams from hyperparams dict into another one, which can be applied as **kwargs, renaming them in process."""
		pythonHyperparams = type(hyperparamsDict)()
		for name, remappedName in cls.hyperparamsNamesRemap.items():
			#print(name, remappedName, name in hyperparamsDict)
			if name in hyperparamsDict:
				pythonHyperparams[remappedName] = hyperparamsDict[name]
				del hyperparamsDict[name]
		return pythonHyperparams


class NotAModelException(ValueError):
	"""Means that model is not a valid model"""


class UnsupportedFormatException(ValueError):
	"""Means that model cannot operate this format"""


def hyperparamsTypeCoerce(hyperparams: Dict[str, Union[int, str, float]]) -> None:
	if "max_depth" in hyperparams:
		hyperparams["max_depth"] = int(hyperparams["max_depth"])
	if "n_estimators" in hyperparams:
		hyperparams["n_estimators"] = int(hyperparams["n_estimators"])
	if "num_boost_round" in hyperparams:
		hyperparams["num_boost_round"] = int(hyperparams["num_boost_round"])
	return hyperparams