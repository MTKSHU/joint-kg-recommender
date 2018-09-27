import utility.utils as metrics
from utility.parser import parse_args

from utility.load_data import *
import multiprocessing

cores = multiprocessing.cpu_count() // 4

args = parse_args()
Ks = eval(args.Ks)

data_generator = Data(path=args.path + args.dataset, batch_size=args.batch_size)
USR_NUM, ITEM_NUM = data_generator.n_users, data_generator.n_items
N_TRAIN, N_TEST = data_generator.n_train, data_generator.n_test
BATCH_SIZE = args.batch_size

def test_one_user(x):
    # user u's ratings for user u
    rating = x[0]
    #uid
    u = x[1]
    #user u's items in the training set
    try:
        training_items = data_generator.train_items[u]
    except Exception:
        training_items = []
    #user u's items in the test set
    user_pos_test = data_generator.test_set[u]

    all_items = set(range(ITEM_NUM))

    test_items = list(all_items - set(training_items))
    item_score = []
    for i in test_items:
        item_score.append((i, rating[i]))

    item_score = sorted(item_score, key=lambda x: x[1])
    item_score.reverse()
    item_sort = [x[0] for x in item_score]

    r = []
    for i in item_sort:
        if i in user_pos_test:
            r.append(1)
        else:
            r.append(0)

    avg_pre, recall, ndcg = [], [], []

    for K in Ks:
        avg_pre.append(metrics.average_precision(r, K))
        recall.append(metrics.recall_at_k(r, K, len(user_pos_test)))
        ndcg.append(metrics.ndcg_at_k(r, K))

    return np.array(recall + avg_pre + ndcg)


def test(sess, model, users_to_test):
    result = np.array([0.] * 15)
    pool = multiprocessing.Pool(cores)


    u_batch_size = BATCH_SIZE * 10
    i_batch_size = BATCH_SIZE

    #all users needed to test
    test_users = users_to_test
    test_user_num = len(test_users)
    index = 0
    while True:
        if index >= test_user_num:
            break
        user_batch = test_users[index:index + u_batch_size]
        index += u_batch_size
        FLAG = False
        if len(user_batch) < u_batch_size:
            user_batch += [user_batch[-1]] * (u_batch_size - len(user_batch))
            user_batch_len = len(user_batch)
            FLAG = True


        user_batch_rating = np.zeros(shape=(len(user_batch), ITEM_NUM))
        n_item_batch = N_TEST// i_batch_size + 1

        for idx in range(n_item_batch):

            start = idx * i_batch_size
            end = min((idx+1)* i_batch_size, ITEM_NUM)
            item_batch = range(start, end)

            if args.model_type not in ['sgraphcf_star', 'gcmc']:
                user_item_batch_rating = sess.run(model.batch_ratings, {model.users: user_batch,
                                                                        model.pos_items: item_batch})
            else:
                user_item_batch_rating = sess.run(model.batch_ratings, {model.users: user_batch,
                                                                        model.pos_items: item_batch,
                                                                        model.node_dropout: [0.]*len(eval(args.layer_size)),
                                                                        model.mess_dropout: [0.]*len(eval(args.layer_size))})
            user_batch_rating[:,start:end] = user_item_batch_rating

        # user_batch_rating = sess.run(model.all_ratings, {model.users: user_batch})
        user_batch_rating_uid = zip(user_batch_rating, user_batch)
        batch_result = pool.map(test_one_user, user_batch_rating_uid)

        if FLAG == True:
            batch_result = batch_result[:user_batch_len]
        for re in batch_result:
            result += re

    pool.close()
    ret = result / test_user_num
    ret = list(ret)
    return ret