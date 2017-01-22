"""A really stupid dumb uniform grid spec with only ranges/boxes available and separate messing with params in the spec and out of it"""

from UniOpt.core.Spec import *
import scipy.stats

defaultGridSpec = {
	"colsample_bytree": HyperparamDefinition(float, scipy.stats.beta(a=10, b=1)),
	"learning_rate": HyperparamDefinition(float, scipy.stats.uniform(loc=0.01, scale=0.44)),
	"max_depth": HyperparamDefinition(int, scipy.stats.uniform(loc=2, scale=43)),  # discrete
	"num_boost_round": HyperparamDefinition(int, scipy.stats.uniform(loc=4, scale=41)),  # discrete
	"min_child_weight": HyperparamDefinition(float, scipy.stats.expon(loc=0, scale=55)),  # often called gamma
	"min_split_loss": HyperparamDefinition(float, scipy.stats.expon(loc=0, scale=12)),
	"reg_alpha": HyperparamDefinition(float, scipy.stats.expon(loc=0, scale=150)),
	"subsample": HyperparamDefinition(float, scipy.stats.beta(a=10, b=1)),
	#'booster': 'dart',
	#"sample_type": ("uniform", "weighted"),
	#"normalize_type": ("tree", "forest"),
	#"rate_drop": HyperparamDefinition(float, scipy.stats.uniform(loc=0., scale=1.)),
	#"skip_drop": HyperparamDefinition(float, scipy.stats.uniform(loc=0., scale=1.)),
}