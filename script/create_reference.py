from itertools import groupby
import os
import sys
import re
import argparse
import unicodedata
import glob
import json
import math
import pickle
import pprint
import sklearn
import hashlib
import numpy as np
from scipy import linalg
from sklearn import mixture
from sklearn.neighbors import LocalOutlierFactor
from scipy import stats

parser = argparse.ArgumentParser(
    description='Categorize audios or detect category.')
parser.add_argument('--task', default='categorize',
                    help='task (categorize/detect)')
parser.add_argument('--vector', default='type1',
                    help='task (type1/type2/type3/type4/type5/type6)')
parser.add_argument('--normalize', action='store_true')
parser.add_argument('--max_iter', type=int, default=100)
parser.add_argument('--max_category', type=int, default=32)
parser.add_argument('--source', default='')
parser.add_argument('--outlier_ratio', type=float, default='0.1')
parser.add_argument('--neighbor_ratio', type=float, default='0.03')
parser.add_argument('--covariance_regularization', type=float, default=1e-3)
parser.add_argument('--precomputed_distance_matrix',
                    default='./distance_matrix.json')
parser.add_argument('--detection_preparation',
                    default='./detection_preparation.json')
parser.add_argument('--analysis_data',
                    default='./resource/analysis_data')
parser.add_argument('--genre_regex', default='.*')
args = parser.parse_args()

pp = pprint.PrettyPrinter(indent=2, stream=sys.stderr)

# http://aidiary.hatenablog.com/entry/20121014/1350211413
# 行列式がはしょられているので注意


def kl_div(mu1, S1, mu2, S2):
    """正規分布間のカルバック・ライブラー情報量"""
    # 逆行列を計算
    # try:
    #     invS1 = np.linalg.inv(S1)
    # except np.linalg.linalg.LinAlgError:
    #     raise
    try:
        invS2 = np.linalg.inv(S2)
    except np.linalg.linalg.LinAlgError:
        raise

    # KL Divergenceを計算
    t1 = np.sum(np.diag(np.dot(invS2, S1)))
    t2 = (mu2 - mu1).transpose()
    t3 = mu2 - mu1
    return 0.5 * (t1 + np.dot(np.dot(t2, invS2), t3) - mu1.size)


def jensen_shannon_distance(mean1, covariance1, mean2, covariance2):
    return max(0, 0.5 * (kl_div(mean1, covariance1, mean2, covariance2) +
                         kl_div(mean2, covariance2, mean1, covariance1))) ** 0.5


def sample_data():
    # Number of samples per component
    n_samples = 500
    # Generate random sample, two components
    np.random.seed(0)
    C = np.array([[0., -0.1], [1.7, .4]])
    X = np.r_[np.dot(np.random.randn(n_samples, 2), C),
              .7 * np.random.randn(n_samples, 2) + np.array([-6, 3])]
    return X


def band_to_vec(band, loudness):
    return [
        band['loudness'] - loudness,
        band['loudness_range'],
        band['mid_to_side_loudness'] - loudness,
        band['mid_to_side_loudness_range']
    ]


def band_to_mean_vec(band, loudness):
    return [
        band['mid_mean'] - loudness,
        # - band['mid_mean'] covarianceの計算方法を整合していないといけないので,
        band['side_mean'] - loudness,
    ]


def regularize(m, r):
    return m + np.diag(np.ones(m.shape[0]) * r)


def load_audio_data(path):
    f = open(path, 'r')
    parsed = json.load(f)
    loudness = parsed['loudness']
    if args.vector == 'type1':
        return np.array(list(map(lambda band: band_to_vec(band, loudness), parsed['bands']))).flatten()
    if args.vector == 'type2':
        cov = np.array(parsed['covariance'])
        log_cov = linalg.logm(cov)
        return log_cov.flatten()
    if args.vector == 'type3':
        means = np.array(list(map(lambda band: band_to_mean_vec(
            band, loudness), parsed['bands']))).flatten()
        cov = np.array(parsed['covariance'])
        log_cov = linalg.logm(cov)
        return np.concatenate((means, log_cov.flatten()))
    if args.vector == 'type4':
        means = np.array(list(map(lambda band: band_to_mean_vec(
            band, loudness), parsed['bands']))).flatten()
        return means
    if args.vector == 'type5':
        means = np.array(
            list(map(lambda band: band['loudness'] - loudness, parsed['bands']))).flatten()
        return means
    if args.vector == 'type6':
        cov = np.array(parsed['covariance_short'])
        log_cov = linalg.logm(cov)
        return log_cov.flatten()
    if args.vector == 'type7':
        means = np.array(list(map(lambda band: band_to_mean_vec(
            band, loudness), parsed['bands_short']))).flatten()
        cov = np.array(parsed['covariance_short'])
        log_cov = linalg.logm(cov)
        return np.concatenate((means, log_cov.flatten()))


def load_audio_data2(path):
    f = open(path, 'r')
    parsed = json.load(f)
    loudness = parsed['loudness']
    means = np.array(list(map(lambda band: band_to_mean_vec(
        band, loudness), parsed['bands_short']))).flatten()
    cov = np.array(parsed['covariance_short'])
    # deci = []
    # for i, a in enumerate(cov):
    # 	if i % 2 == 0:
    # 		deci2 = []
    # 		for j, b in enumerate(a):
    # 			if j % 2 == 0:
    # 				deci2.append(b)
    # 		deci.append(deci2)
    # cov = np.array(deci)
    return {
        'mean': means,
        'covariance': cov
    }


def normalize_data(data):
    mean = np.mean(np.transpose(data), 1)
    stddev = np.std(np.transpose(data), 1)
    pp.pprint(mean)
    pp.pprint(stddev)
    return (data - mean) / stddev


def audio_paths():
    return sorted(glob.glob(args.analysis_data + '/**/*.json', recursive=True))


def audio_data():
    files = audio_paths()
    data = np.array(list(map(load_audio_data, files)))
    pp.pprint(data)
    if args.normalize:
        data = normalize_data(data)
    return data


def categorize():
    pp.pprint('categorize')
    X = audio_data()
    dpgmm = mixture.BayesianGaussianMixture(
        n_components=args.max_category,
        covariance_type='full',
        max_iter=args.max_iter,
        init_params="kmeans",
        tol=1e-3,
        reg_covar=1e-6).fit(X)
    pp.pprint(X)

    pp.pprint(dpgmm.predict(X).size)
    pp.pprint(np.unique(dpgmm.predict(X)).size)
    pp.pprint(dpgmm.predict(X))

    paths = audio_paths()
    category_to_audio_ids = {}
    for audio_id, category in enumerate(dpgmm.predict(X)):
        category_to_audio_ids.setdefault(category, [])
        category_to_audio_ids[category].append(audio_id)
    for category, audio_ids in category_to_audio_ids.items():
        for audio_id in audio_ids:
            pp.pprint(category, paths[audio_id])

    pp.pprint(dpgmm.means_)
    pp.pprint(dpgmm.covariances_)
    pp.pprint(dpgmm.n_iter_)
    pp.pprint(dpgmm.lower_bound_)
    pp.pprint(dpgmm.converged_)
    pp.pprint(dpgmm.weight_concentration_prior_)
    pp.pprint(dpgmm.weight_concentration_)
    # pp.pprint(pickle.dumps(dpgmm))


def precomputed_distance_matrix(data):
    result = np.zeros((data.shape[0], data.shape[0]))
    for i, a in enumerate(data):
        pp.pprint('precomputed_distance_matrix %d / %d' % (i, data.shape[0]))
        for j, b in enumerate(data):
            if i < j:
                dis = jensen_shannon_distance(
                    a['mean'], regularize(
                        a['covariance'], args.covariance_regularization),
                    b['mean'], regularize(
                        b['covariance'], args.covariance_regularization)
                )
                result[i, j] = dis
                result[j, i] = dis
    return result


def distance_vec_to_score(vec, k, skip_first):
    score = 0
    count = k
    if skip_first:
        count += 1
    for i, a in enumerate(sorted(vec)):
        if i >= count:
            break
        else:
            score = max(score, a)  # score += a
    return score  # / k


def calc_q(vec, x):
    for i, a in enumerate(vec):
        if x < a:
            return i / len(vec)
    return 1


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def prepare_for_detection():
    pp.pprint('prepare_for_detection')
    paths = audio_paths()

    filtered_paths = []
    regex = re.compile(args.genre_regex, re.IGNORECASE)
    for path in paths:
        f = open(path, 'r')
        parsed = json.load(f)
        genre = parsed.get('ffprobe', {}).get(
            'format', {}).get('tags', {}).get('genre', '')
        genre = unicodedata.normalize('NFKC', genre)
        if regex.search(genre):
            pp.pprint(genre)
            filtered_paths.append(path)

    pp.pprint(filtered_paths)
    data = np.array(list(map(load_audio_data2, filtered_paths)))

    distance_matrix = precomputed_distance_matrix(data)

    output = {
        'covariance_regularization': args.covariance_regularization,
        'paths': filtered_paths,
        'distance_matrix': distance_matrix,
    }
    print(json.dumps(output, indent=2, sort_keys=True, cls=NumpyEncoder))


def detect():
    pp.pprint('detect')

    f = open(args.detection_preparation, 'r')
    detection_preparation = json.load(f)

    source_data = load_audio_data2(args.source)
    paths = detection_preparation['paths']  # audio_paths()
    data = np.array(list(map(load_audio_data2, paths)))

    ranking = []
    distance_vec = []
    for audio_id, d in enumerate(data):
        try:
            dis = jensen_shannon_distance(source_data['mean'], regularize(source_data['covariance'], detection_preparation['covariance_regularization']),
                                          d['mean'], regularize(d['covariance'], detection_preparation['covariance_regularization']))
            ranking.append([dis, audio_id])
            distance_vec.append(dis)
        except np.linalg.linalg.LinAlgError:
            pp.pprint('error', paths[audio_id])
    ranking = sorted(ranking)
    for row in ranking:
        pp.pprint('%.2f %s' % (row[0], paths[row[1]]))

    neighbor_count = math.ceil(len(paths) * args.neighbor_ratio)
    # precomputed_distance_matrix(data)
    distance_matrix = detection_preparation['distance_matrix']
    scores = []
    for vec in distance_matrix:
        scores.append(distance_vec_to_score(vec, neighbor_count, True))
    scores = sorted(scores)

    score = distance_vec_to_score(distance_vec, neighbor_count, False)
    quantile = calc_q(scores, score)
    pp.pprint('q: %.3f' % (quantile))

    output = {
        'outlierness_quantile': quantile,
        'ranking': [
            {
                'path': paths[ranking[0][1]],

            }
        ]
    }
    print(json.dumps(output, indent=2, sort_keys=True, cls=NumpyEncoder))


def audio_data2_to_vec(data2):
    dim = data2['mean'].size
    vec = np.zeros((dim + dim * dim))
    for k in range(dim):
        vec[k] = data2['mean'][k]
        for m in range(dim):
            vec[dim + k * dim + m] = data2['covariance'][k, m]
    return vec


def get_vec_index(vec1, data2_dict, data2_dict_key_precision):
    key = round(vec1[0], data2_dict_key_precision)
    if key in data2_dict:
        indicies = data2_dict[key]
        if len(indicies) == 1:
            return indicies[0]
    return -1


def detect2_metric_func(vec1, vec2, dim=0, distance_matrix=0, data2_dict=0, data2_dict_key_precision=0):
    i1 = get_vec_index(vec1, data2_dict, data2_dict_key_precision)
    i2 = get_vec_index(vec2, data2_dict, data2_dict_key_precision)
    if i1 >= 0 and i2 >= 0:
        return distance_matrix[i1][i2]
    else:
        return jensen_shannon_distance(
            vec1[0:dim], regularize(
                vec1[dim:].reshape(dim, dim), args.covariance_regularization),
            vec2[0:dim], regularize(
                vec2[dim:].reshape(dim, dim), args.covariance_regularization)
        )


def detect2():
    pp.pprint('detect2')

    f = open(args.detection_preparation, 'r')
    detection_preparation = json.load(f)

    source_data = load_audio_data2(args.source)
    source_data2 = audio_data2_to_vec(source_data)
    paths = detection_preparation['paths']
    data = np.array(list(map(load_audio_data2, paths)))

    data2_dict_key_precision = 7

    dim = data[0]['mean'].size
    data2 = np.zeros((len(paths), dim + dim * dim))
    data2_dict = {}
    for i, d in enumerate(data):
        v = audio_data2_to_vec(d)
        for k in range(v.size):
            data2[i, k] = v[k]
        key = round(data2[i, 0], data2_dict_key_precision)
        data2_dict.setdefault(key, [])
        data2_dict[key].append(i)

    lof_options = {
        'n_neighbors': math.ceil(args.neighbor_ratio * len(paths)),
        'metric': detect2_metric_func,
        'algorithm': 'brute',
        'n_jobs': 1,
        'metric_params': {
            'dim': dim,
            'distance_matrix': detection_preparation['distance_matrix'],
            'data2_dict': data2_dict,
            'data2_dict_key_precision': data2_dict_key_precision,
        }
    }
    options_and_data = {
        'lof_options': lof_options,
        'data2': data2,
    }

    os.makedirs('/tmp/phaselimiter/create_reference', exist_ok=True)
    cache_path = '/tmp/phaselimiter/create_reference/' + \
        hashlib.md5(pickle.dumps(options_and_data)).hexdigest()
    pp.pprint(cache_path)
    if os.path.isfile(cache_path):
        with open(cache_path, mode='rb') as f:
            clf = pickle.load(f)
    else:
        clf = LocalOutlierFactor(**lof_options)
        clf.fit_predict(data2)
        with open(cache_path, mode='wb') as f:
            pickle.dump(clf, f)

    source_lof = -clf._decision_function(source_data2.reshape(1, -1))[0]
    border_lof = - \
        stats.scoreatpercentile(
            clf.negative_outlier_factor_, 100 * args.outlier_ratio)

    ranking = []
    distance_vec = []
    for audio_id, d in enumerate(data):
        lof = -clf.negative_outlier_factor_[audio_id]
        if lof <= border_lof:
            dis = jensen_shannon_distance(source_data['mean'], regularize(source_data['covariance'], detection_preparation['covariance_regularization']),
                                          d['mean'], regularize(d['covariance'], detection_preparation['covariance_regularization']))
            ranking.append([dis, audio_id, lof])
            distance_vec.append(dis)
    ranking = sorted(ranking)
    for row in ranking:
        pp.pprint('%.2f %.2f %s' % (row[0], row[2], paths[row[1]]))

    output = {
        'border_lof': border_lof,
        'lof': source_lof,
        'sound_quality': 1 / (1 + abs(source_lof - 1) / (border_lof - 1)),
        'ranking': [
            {
                'path': paths[ranking[0][1]],
                'lof': ranking[0][2],
            }
        ]
    }
    print(json.dumps(output, indent=2, sort_keys=True, cls=NumpyEncoder))


if args.task == 'categorize':
    categorize()
elif args.task == 'detect':
    detect()
elif args.task == 'detect2':
    detect2()
elif args.task == 'prepare_for_detection':
    prepare_for_detection()

pp.pprint('finished')
