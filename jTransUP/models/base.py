import time
import gflags
import torch
from copy import deepcopy
from functools import reduce

from jTransUP.data import load_rating_data
from jTransUP.utils.data import MakeTrainIterator, MakeEvalIterator
import jTransUP.models.transUP as transup
import jTransUP.models.bprmf as bprmf

def get_data_manager(dataset):
    # Select data format.
    if dataset in ["ml1m","dbbook2014"]:
        data_manager = load_rating_data
    else:
        raise NotImplementedError

    return data_manager

def load_data(FLAGS, logger):
    data_manager = get_data_manager(FLAGS.dataset)

    trainDict, testDict, validDict, allRatingDict, user_total, item_total, trainTotal, testTotal, validTotal = data_manager.load_data(FLAGS.data_path+FLAGS.dataset, logger=logger)

    train_iter = MakeTrainIterator(trainDict, item_total, FLAGS.batch_size,negtive_samples=FLAGS.negtive_samples, allRatingDict=allRatingDict)

    test_iter = MakeEvalIterator(testDict, item_total, FLAGS.batch_size, allRatingDict=allRatingDict)

    valid_iter = None
    if validDict is not None:
        valid_iter = MakeEvalIterator(validDict, item_total, FLAGS.batch_size, allRatingDict=allRatingDict)

    return train_iter, test_iter, valid_iter, user_total, item_total, trainTotal, testTotal, validTotal, testDict, validDict

def get_flags():
    gflags.DEFINE_enum("model_type", "transup", ["transup", "bprmf"], "")
    gflags.DEFINE_enum("dataset", "ml1m", ["ml1m", "dbbook2014"], "")
    gflags.DEFINE_float("learning_rate", 0.001, "Used in optimizer.")
    gflags.DEFINE_integer(
        "early_stopping_steps_to_wait",
        70000,
        "How many times will lr decrease? If set to 0, it remains constant.")
    gflags.DEFINE_bool(
        "L1_flag",
        False,
        "If set to True, use L1 distance as dissimilarity; else, use L2.")
    gflags.DEFINE_float("l2_lambda", 1e-5, "")
    gflags.DEFINE_integer("embedding_size", 64, ".")
    gflags.DEFINE_integer("negtive_samples", 1, ".")
    gflags.DEFINE_integer("batch_size", 512, "Minibatch size.")
    gflags.DEFINE_enum("optimizer_type", "SGD", ["Adam", "SGD", "Adagrad", "Rmsprop"], "")
    gflags.DEFINE_float("learning_rate_decay_when_no_progress", 0.5,
                        "Used in optimizer. Decay the LR by this much every epoch steps if a new best has not been set in the last epoch.")

    gflags.DEFINE_integer(
        "eval_interval_steps",
        14000,
        "Evaluate at this interval in each epoch.")
    gflags.DEFINE_integer(
        "training_steps",
        1400000,
        "Stop training after this point.")
    gflags.DEFINE_enum("loss_type", "bpr", ["bpr", "margin"], "")
    gflags.DEFINE_float("clipping_max_value", 5.0, "")
    gflags.DEFINE_float("margin", 1.0, "Used in margin loss.")
    gflags.DEFINE_float("momentum", 0.9, "The momentum of the optimizer.")
    gflags.DEFINE_integer("seed", 0, "Fix the random seed. Except for 0, which means no setting of random seed.")
    gflags.DEFINE_integer("topn", 10, "")
    gflags.DEFINE_integer("num_preferences", 4, "")

    gflags.DEFINE_string("experiment_name", None, "")
    gflags.DEFINE_string("data_path", None, "")
    gflags.DEFINE_string("log_path", None, "")
    gflags.DEFINE_enum("log_level", "debug", ["debug", "info"], "")
    gflags.DEFINE_string(
        "ckpt_path", None, "Where to save/load checkpoints. If not set, the same as log_path")

    gflags.DEFINE_boolean(
        "has_visualization",
        True,
        "if set True, use visdom for visualization.")
    # todo: only eval when no train.dat when load data
    gflags.DEFINE_boolean(
        "eval_only_mode",
        False,
        "If set, a checkpoint is loaded and a forward pass is done to get the predicted candidates."
        "Requirements: Must specify load_experiment_name.")
    gflags.DEFINE_string("load_experiment_name", None, "")

def flag_defaults(FLAGS):

    if not FLAGS.experiment_name:
        timestamp = str(int(time.time()))
        FLAGS.experiment_name = "{}-{}".format(
            FLAGS.dataset,
            timestamp,
        )

    if not FLAGS.data_path:
        FLAGS.data_path = "../datasets/"

    if not FLAGS.log_path:
        FLAGS.log_path = "../log/"

    if not FLAGS.ckpt_path:
        FLAGS.ckpt_path = FLAGS.log_path

    if FLAGS.eval_only_mode and not FLAGS.load_experiment_name:
        FLAGS.load_experiment_name = FLAGS.experiment_name
    
    if FLAGS.seed != 0:
        torch.manual_seed(FLAGS.seed)


def init_model(
        FLAGS,
        user_total,
        item_total,
        logger):
    # Choose model.
    logger.info("Building model.")
    if FLAGS.model_type == "transup":
        build_model = transup.build_model
    elif FLAGS.model_type == "bprmf":
        build_model = bprmf.build_model
    else:
        raise NotImplementedError

    model = build_model(FLAGS, user_total, item_total)

    # Print model size.
    logger.info("Architecture: {}".format(model))

    total_params = sum([reduce(lambda x, y: x * y, w.size(), 1.0)
                        for w in model.parameters()])
    logger.info("Total params: {}".format(total_params))

    return model