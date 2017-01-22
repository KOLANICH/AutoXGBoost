import types
from typing import Dict
from pathlib import Path
import shutil
import os

try:
	import simplejson as json
except ImportError:
	import json


class IterableModule(types.ModuleType):
	__all__ = None

	def __iter__(self):
		for pn in self.__class__.__all__:
			yield getattr(self, pn)


def serializeDict(fileName: Path, dic: Dict[str, object]) -> None:
	os.makedirs(str(fileName.parent), exist_ok=True)
	with fileName.open("wt", encoding="utf-8") as f:
		json.dump(dic, f, indent="\t")


def parseDict(fileName: Path) -> Dict:
	if not fileName.exists():
		return {}
	with fileName.open("rt", encoding="utf-8") as f:
		return json.load(f)


def resolveAvailablePath(fileName: str):
	p = shutil.which(fileName)
	if p:
		return Path(p).resolve().absolute()


def explainItemUsingSHAP(shapValues: "pandas.DataFrame", thresholdRatio=50.):
	res = []
	for idx, el in shapValues.iterrows():
		normConst = el.sum()
		el /= normConst
		shapValsSignificance = el.abs().sort_values(ascending=False)
		minThresh = shapValsSignificance[0] / thresholdRatio
		selector = shapValsSignificance > minThresh
		shapValsSignificance = shapValsSignificance[selector]
		significantShaps = el[shapValsSignificance.index]
		resDict = significantShaps.to_dict()
		resDict["$other"] = el[~selector].sum()
		res.append(resDict)
	return res