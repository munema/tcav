import os
from scipy.stats import ttest_ind
import numpy as np
from tcav.utils import pickle_load

def get_cav_accuracy(results):
  # helper function, returns if this is a random concept
  def is_random_concept(concept):
    return 'random500_' in concept

  # print class, it will be the same for all
  # print("Class =", results[0]['target_class'])

  # prepare data
  # dict with keys of concepts containing dict with bottlenecks
  result_summary = {}

  # random
  random_accuracy = {}

  for result in results:
    if result['cav_concept'] not in result_summary:
      result_summary[result['cav_concept']] = {}

    if result['bottleneck'] not in result_summary[result['cav_concept']]:
      result_summary[result['cav_concept']][result['bottleneck']] = []

    result_summary[result['cav_concept']][result['bottleneck']].append(result)

    # store random
    if is_random_concept(result['cav_concept']):
      if result['bottleneck'] not in random_accuracy:
        random_accuracy[result['bottleneck']] = []

      random_accuracy[result['bottleneck']].append(result['cav_accuracies'])

  cav_accuracy = {}

  # print concepts and classes with indentation
  for concept in result_summary:
    # if not random
    if not is_random_concept(concept):
      if concept not in cav_accuracy:
        cav_accuracy[concept] = {}
      for bottleneck in result_summary[concept]:
        if bottleneck not in cav_accuracy[concept]:
          cav_accuracy[concept][bottleneck] = {}
        i_ups = [item['cav_accuracies'] for item in result_summary[concept][bottleneck]]
        cav_accuracy[concept][bottleneck] = i_ups

  # add random cav_accuracy
  cav_accuracy['random'] = {}
  for bottleneck in random_accuracy:
    cav_accuracy['random'][bottleneck] = random_accuracy[bottleneck]

  return cav_accuracy

def get_score_dist(results):
  # helper function, returns if this is a random concept
  def is_random_concept(concept):
    return 'random500_' in concept

  # print class, it will be the same for all
  # print("Class =", results[0]['target_class'])

  # prepare data
  # dict with keys of concepts containing dict with bottlenecks
  result_summary = {}
  # random
  random_i_ups = {}
  for result in results:
    if result['cav_concept'] not in result_summary:
      result_summary[result['cav_concept']] = {}
    if result['bottleneck'] not in result_summary[result['cav_concept']]:
      result_summary[result['cav_concept']][result['bottleneck']] = []
    result_summary[result['cav_concept']][result['bottleneck']].append(result)

    # store random
    if is_random_concept(result['cav_concept']):
      if result['bottleneck'] not in random_i_ups:
        random_i_ups[result['bottleneck']] = []
      random_i_ups[result['bottleneck']].append(result['i_up'])

  dist = {}
  # print concepts and classes with indentation
  for concept in result_summary:
    # if not random
    if not is_random_concept(concept):
      if concept not in dist:
        dist[concept] = {}
      for bottleneck in result_summary[concept]:
        if bottleneck not in dist[concept]:
          dist[concept][bottleneck] = {}
        i_ups = [item['i_up'] for item in result_summary[concept][bottleneck]]
        dist[concept][bottleneck] = i_ups
  # add random dist
  dist['random'] = {}
  for bottleneck in random_i_ups:
    dist['random'][bottleneck] = random_i_ups[bottleneck]

  return dist

# CAVの平均
def get_cav_mean(path, bottleneck, concept):
    cnt = 0
    for cav_dir in os.listdir(path):
        if cav_dir.split(':')[0] != 'cav':
            continue
        t, c, nc, b = cav_dir.split(':')
        b = b.split('.')[0]
        if b == bottleneck and c == concept:
            if cnt == 0:
                cav_values = np.array(pickle_load(path + '/' + cav_dir)['cavs'][0])
            else:
                cav_values += np.array(pickle_load(path + '/' + cav_dir)['cavs'][0])
            cnt += 1
    cav_values /= cnt
    return cav_values

# CAV
def get_cav(path, bottleneck, concept):
  cav_lst = []
  for cav_dir in os.listdir(path):
      if cav_dir.split(':')[0] != 'cav':
          continue
      t, c, nc, b = cav_dir.split(':')
      b = b.split('.')[0]
      if b == bottleneck and c == concept:
        cav_values = np.array(pickle_load(path + '/' + cav_dir)['cavs'][0])
        cav_lst.append(cav_values)
  return cav_lst

def get_random_cav(path, bottleneck):
  cav_lst = []
  for cav_dir in os.listdir(path):
      if cav_dir.split(':')[0] != 'cav':
          continue
      t, c, nc, b = cav_dir.split(':')
      b = b.split('.')[0]
      if b == bottleneck and 'random500_' in c:
        cav_values = np.array(pickle_load(path + '/' + cav_dir)['cavs'][0])
        cav_lst.append(cav_values)
  return cav_lst

# 真のCAVの平均を得る
def get_true_cav_mean(path, bottleneck, concept):
    cnt = 0
    for cav_dir in os.listdir(path):
        if cav_dir.split(':')[0] != 'cav-true':
            continue
        t, en, c, b = cav_dir.split(':')
        if b == bottleneck and c == concept:
            if cnt == 0:
                cav_values = np.array(pickle_load(path + '/' + cav_dir))
            else:
                cav_values += np.array(pickle_load(path + '/' + cav_dir))
            cnt += 1
    cav_values /= cnt
    return cav_values


# sensitivityを得る
def get_sensitivity(results, concept, bottleneck):
    s_lst = []
    for result in results:
        if result['cav_concept'] == concept and result['bottleneck'] == bottleneck:
            s_lst.append(result['val_directional_dirs'])
    return np.array(s_lst)

# scoreを得る
def get_score(results, concept, bottleneck):
    s_lst = []
    for result in results:
        if result['cav_concept'] == concept and result['bottleneck'] == bottleneck:
            s_lst.append(result['i_up'])
    return np.array(s_lst)


# 予測値を得る
def get_predict(path, target):
    for dir in os.listdir(path):
        if dir.split(':')[0] != 'predict':
            continue
        _, t = dir.split(':')
        if t == target:
            values = np.array(pickle_load(path + '/' + dir))
    return values

# 勾配を得る
def get_grad(path, bottleneck, target):
    for dir in os.listdir(path):
        if dir.split(':')[0] != 'grad':
            continue
        _, b,t = dir.split(':')
        if b == bottleneck and t == target:
            values = np.array(pickle_load(path + '/' + dir))
    return values

# logitの勾配を得る
def get_logit_grad(cavs_path, bottleneck,target):
    pred = get_predict(cavs_path, target)
    grad = get_grad(cavs_path, bottleneck,target)
    cnt = 0
    for cav_dir in os.listdir(cavs_path):
        if cav_dir.split(':')[0] != 'grad':
          continue
        _, b,t = cav_dir.split(':')
        if b == bottleneck and t == target:
          s = np.array([-pred[i]*grad[i] for i in range(len(pred))])
    return s

# cos類似度
def cos_sim(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))