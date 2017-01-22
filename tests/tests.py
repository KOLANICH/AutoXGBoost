#!/usr/bin/env python3
import sys
from pathlib import Path
import unittest
thisDir=Path(__file__).parent.absolute()
sys.path.insert(0, str(thisDir.parent))
modelsPrefix=thisDir / "models"

import numpy as np
import pandas
from pandas.testing import assert_frame_equal, assert_series_equal
import AutoXGBoost as axgb
from AutoXGBoost import *
import UniOpt

from os import urandom
import struct
from pprint import pprint
from collections import OrderedDict

vectorCount=100
vectorLen=2

def randomVector():
	np.random.rand(vectorLen)

def randomDataset(*args, withWeight=True, **kwargs):
	ds=np.random.rand(vectorCount, vectorLen+int(withWeight))
	cns=["A"+str(a) for a in range(vectorLen)]
	spec={cn:"numerical" for cn in cns}
	
	if withWeight:
		cns.append("w")
		spec["w"] = "weight"
	
	ds=pandas.DataFrame(ds, columns=cns)
	return axgb.AutoXGBoost(spec, ds, prefix=modelsPrefix, *args, **kwargs)

class SimpleTests(unittest.TestCase):
	@classmethod
	def setUpClass(cls):
		cls.withWeight=True
		cls.ds=randomDataset(withWeight=cls.withWeight)
	
	def testTrain(self):
		columnsCountStr=str(len(self.__class__.ds.columns))
		with self.subTest("getIssues uninitialized dict"):
			self.assertEqual(self.__class__.ds.getIssues(), {'No data': False, 'stop columns processed': {}, 'HP unknown': set(self.__class__.ds.columns), 'HP known, but no models': set(), 'not scored': set()})
		
		self.__class__.ds.loadHyperparams()
		for cn in self.__class__.ds.groups["numerical"]:
			self.assertIn(cn, self.__class__.ds.bestHyperparams)
		
		with self.subTest("getIssues hps loaded dict"):
			self.assertEqual(self.__class__.ds.getIssues(), {'No data': False, 'stop columns processed': {}, 'HP unknown': set(), 'HP known, but no models': set(self.__class__.ds.columns), 'not scored': set()})
		
		self.__class__.ds.trainModels()
		for cn in self.__class__.ds.bestHyperparams:
			self.assertIn(cn, self.__class__.ds.models)
		
		with self.subTest("getIssues models trained dict"):
			self.assertEqual(self.__class__.ds.getIssues(), {'No data': False, 'stop columns processed': {}, 'HP unknown': set(), 'HP known, but no models': set(), 'not scored': set(self.__class__.ds.columns)})
		
		self.__class__.ds.scoreModels()
		for cn in self.__class__.ds.models:
			self.assertIn(cn, self.__class__.ds.scores)
		
		okDict={'No data': False, 'stop columns processed': {}, 'HP unknown': set(), 'HP known, but no models': set(), 'not scored': set()}
		with self.subTest("getIssues models scores known dict"):
			self.assertEqual(self.__class__.ds.getIssues(), okDict)
		
		with self.subTest("getIssues models scores known repr"):
			r = repr(self.__class__.ds)
			firstPart = self.__class__.ds.__class__.__name__+"< columns: "+columnsCountStr+", "
			lastPart = " | OK >"
			self.assertEqual(r[:len(firstPart)], firstPart)
			self.assertEqual(r[-len(lastPart):], lastPart)
			middlePartActual = r[len(firstPart): -len(lastPart)]
			
			if self.__class__.withWeight:
				self.assertTrue("weight: 1" in middlePartActual)
			self.assertTrue("numerical: "+columnsCountStr in middlePartActual)
		
		self.assertEqual(self.__class__.ds.getIssues(), okDict)
		
		self.__class__.ds.sortScores()
		self.assertEqual(self.__class__.ds.getIssues(), okDict)
		
		print("ds.scores", self.__class__.ds.scores)
		self.assertEqual(self.__class__.ds.getIssues(), okDict)
		
		self.__class__.ds.saveModels(format=formats.binary)
		print('ds.predict("A1")', self.__class__.ds.predict("A1"))

class SavingDumbTests(unittest.TestCase):
	@classmethod
	def setUpClass(cls):
		cls.ds=randomDataset()
		cls.ds.loadHyperparams()
		cls.ds.trainModels()
		cls.ds.scoreModels(folds=3)
		cls.ds.sortScores()
	
	def testSaveFormat(self):
		for modelClass in models:
			for format in formats:
				with self.subTest(modelClass=modelClass, format=format):
					self.__class__.ds.saveModels(format=format)

class PrepareDatasetForConversionTests(unittest.TestCase):
	@classmethod
	def setUpClass(cls):
		cls.ds=randomDataset()
		cls.ds.loadHyperparams()
		cls.ds.trainModels()
		cls.ds.scoreModels(folds=5)
		cls.ds.sortScores()
		cls.colName="A0"
		cls.model=cls.ds.models[cls.colName]
		cls.tv=cls.ds.prepareCovariates(cls.colName)
		cls.etalonRes=cls.model.predict(cls.tv)

class TestsConversionFromNative(PrepareDatasetForConversionTests):
	def testConversion(self):
		m=self.__class__.model
		for modelClass in models:
			converted=m.convert(modelClass)
			if converted:
				converted.predict(self.__class__.tv)

class DifferrentFormatsTest(PrepareDatasetForConversionTests):
	@classmethod
	def setUpClass(cls):
		super().setUpClass()
		
		cls.colName=next(iter(cls.ds.groups["numerical"]))
		m=cls.ds.models[cls.colName]
		cls.modelsInDifferentFormats={}
		for modelClass in models:
			cls.modelsInDifferentFormats[modelClass]=m.convert(modelClass)

class ConvertingTests(DifferrentFormatsTest):
	def testConversion(self):
		ds=self.__class__.ds
		for frm in models:
			for to in models:
				with self.subTest(frm=frm, to=to):
					self.__class__.modelsInDifferentFormats[frm].convert(to)

class SavingTests(DifferrentFormatsTest):
	def testSavingFromDifferent(self):
		for modelClass in models:
			for format in formats:
				m=self.__class__.modelsInDifferentFormats[modelClass]
				cn=self.__class__.colName
				with self.subTest(modelClass=modelClass, format=format):
					if format.ctor in modelClass.exports.keys():
						m.save(cn, None, format=format)

class LoadingTests(DifferrentFormatsTest):
	@classmethod
	def setUpClass(cls):
		cls.ds=randomDataset()
		cls.ds.loadHyperparams()
		cls.ds.trainModels()
		cls.ds.scoreModels(folds=5)
		cls.ds.sortScores()
		for format in formats:
			cls.ds.saveModels(format=format)
	
	def testLoadFormat(self):
		for modelClass in models:
			for format in formats:
				with self.subTest(modelClass=modelClass, format=format):
					if format in modelClass.imports:
						self.__class__.ds.loadModels(format=format, modelCtor=modelClass)
					else:
						with self.assertRaises(axgb.core.UnsupportedFormatException) as cm:
							self.__class__.ds.loadModels(format=format, modelCtor=modelClass)
						self.assertEqual(cm.exception.args, ('load', format, modelClass))
	
	#@unittest.skip
	def testSaveLoadAutoGuessFormat(self):
		for format in formats:
			if format.ctor:
				with self.subTest(format=format, ctor=format.ctor):
					modelClass=format.ctor
					if format in modelClass.imports:
						self.__class__.ds.loadModels(format=format, modelCtor=modelClass)
					else:
						with self.assertRaises(axgb.core.UnsupportedFormatException) as cm:
							self.__class__.ds.loadModels(format=format, modelCtor=modelClass)
						self.assertEqual(cm.exception.args, ('load', format, modelClass))


class OptimizersTests(unittest.TestCase):
	@classmethod
	def setUpClass(cls):
		cls.ds=randomDataset()
		cls.columnsToFit = {next(iter(cls.ds.groups["numerical"]))}
	
	@unittest.skip
	def testOptimizers(self):
		results={}
		for optimizer in UniOpt:
			with self.subTest(optimizer=optimizer):
				self.__class__.ds.optimizeHyperparams(iters=20, optimizer=optimizer, force=True, columns=self.__class__.columnsToFit)
				self.__class__.ds.trainModels()
				self.__class__.ds.scoreModels()
				results[optimizer]=next(reversed(self.__class__.ds.scores.values()))
		results=OrderedDict(((k.__name__, v) for k,v in sorted(results.items(), key=lambda x: x[1][0])))
		pprint(results)
	
	#@unittest.skip
	def testOptimizer(self):
		"""
		UniOpt.backends.GPyOpt.GPyOpt
		UniOpt.backends.pySOT.PySOT
		"""
		self.__class__.ds.optimizeHyperparams(iters=20, optimizer=UniOpt.GPyOpt, force=True, columns=self.__class__.columnsToFit)

if __name__ == '__main__':
	unittest.main()
