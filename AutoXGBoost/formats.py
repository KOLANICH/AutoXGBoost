from .imports import *
from .utils import IterableModule
from .core.FormatConverter import FormatConverter, OFDT


class Formats(IterableModule):
	__all__ = ("text", "json", "binary", "python")

	class ConverterGetDump(FormatConverter):
		ctor = None

	class _internalXGBoostBoosterFormat(FormatConverter):
		"""Assummed never be dumped into file, used in a python program for conversion only"""

		ext = None

	class text(ConverterGetDump):
		ext = "modeltxt"

	class json(ConverterGetDump):
		ext = "modeljson"

	class binary(FormatConverter):
		ext = "model"
		#ctor=XGBoostModel registered in model

	class python(FormatConverter):
		ext = "py"
		#ctor=pyXGBoostModel #  registered in model


sys.modules[__name__] = Formats(__name__)