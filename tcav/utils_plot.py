"""
Copyright 2018 Google LLC

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from scipy.stats import ttest_ind
import numpy as np
import matplotlib.pyplot as plt


# helper function to output plot and write summary data
def plot_results(results, random_counterpart=None, random_concepts=None, num_random_exp=100,
    min_p_val=0.05, is_bonferroni = False, plot_random = True):
  """Helper function to organize results.
  When run in a notebook, outputs a matplotlib bar plot of the
  TCAV scores for all bottlenecks for each concept, replacing the
  bars with asterisks when the TCAV score is not statistically significant.
  If you ran TCAV with a random_counterpart, supply it here, otherwise supply random_concepts.
  If you get unexpected output, make sure you are using the correct keywords.

  Args:
    results: dictionary of results from TCAV runs.
    random_counterpart: name of the random_counterpart used, if it was used. 
    random_concepts: list of random experiments that were run. 
    num_random_exp: number of random experiments that were run.
    min_p_val: minimum p value for statistical significance
  """

  # helper function, returns if this is a random concept
  def is_random_concept(concept):
    if random_counterpart:
      return random_counterpart == concept
    
    elif random_concepts:
      return concept in random_concepts

    else:
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
    
  # to plot, must massage data again 
  plot_data = {}
  plot_concepts = []
  
  # (変更) ボンフェローニ補正追加
  if is_bonferroni:
    count_concepts = 0
    for concept in result_summary:
      if not is_random_concept(concept):
        count_concepts += 1
    min_p_val /= count_concepts
    
  num_concepts = 0
  
  # print concepts and classes with indentation
  for concept in result_summary:
        
    # if not random
    if not is_random_concept(concept):
      # print(" ", "Concept =", concept)
      plot_concepts.append(concept)
      num_concepts += 1

      for bottleneck in result_summary[concept]:
        i_ups = [item['i_up'] for item in result_summary[concept][bottleneck]]
        
        # Calculate statistical significance
        # print(random_i_ups)
        _, p_val = ttest_ind(random_i_ups[bottleneck], i_ups)
                  
        if bottleneck not in plot_data:
          plot_data[bottleneck] = {'bn_vals': [], 'bn_stds': [], 'significant': []}

        # (変更) 棄却されていないものも表示させる
        if p_val > min_p_val:
          # statistically insignificant
          #plot_data[bottleneck]['bn_vals'].append(0.01)
          #plot_data[bottleneck]['bn_stds'].append(0)
          plot_data[bottleneck]['bn_vals'].append(np.mean(i_ups))
          plot_data[bottleneck]['bn_stds'].append(np.std(i_ups))
          plot_data[bottleneck]['significant'].append(False)
            
        else:
          plot_data[bottleneck]['bn_vals'].append(np.mean(i_ups))
          plot_data[bottleneck]['bn_stds'].append(np.std(i_ups))
          plot_data[bottleneck]['significant'].append(True)

        # print(3 * " ", "Bottleneck =", ("%s. TCAV Score = %.2f (+- %.2f), "
        #     "random was %.2f (+- %.2f). p-val = %.3f (%s)") % (
        #     bottleneck, np.mean(i_ups), np.std(i_ups),
        #     np.mean(random_i_ups[bottleneck]),
        #     np.std(random_i_ups[bottleneck]), p_val,
        #     "not significant" if p_val > min_p_val else "significant"))
        
  # subtract number of random experiments
  # if random_counterpart:
  #   num_concepts = len(result_summary) - 1
  # elif random_concepts:
  #   num_concepts = len(result_summary) - len(random_concepts)
  # else: 
  #   num_concepts = len(result_summary) - num_random_exp
  
  # randomプロット
  if plot_random:
    for bottleneck in random_i_ups:
        plot_data[bottleneck]['bn_vals'].append(np.mean(random_i_ups[bottleneck]))
        plot_data[bottleneck]['bn_stds'].append(np.std(random_i_ups[bottleneck]))
        plot_data[bottleneck]['significant'].append(True)    
    num_concepts += 1
    plot_concepts += ['random']
  num_bottlenecks = len(plot_data)
  bar_width = 0.35
    
  # create location for each bar. scale by an appropriate factor to ensure 
  # the final plot doesn't have any parts overlapping
  index = np.arange(num_concepts) * bar_width * (num_bottlenecks + 1)
  
  
  # matplotlib
  fig, ax = plt.subplots()
    
  # draw all bottlenecks individually
  for i, [bn, vals] in enumerate(plot_data.items()):
    bar = ax.bar(index + i * bar_width, vals['bn_vals'],
        bar_width, yerr=vals['bn_stds'], label=bn)
    # draw stars to mark bars that are stastically insignificant to 
    # show them as different from others
    # for j, significant in enumerate(vals['significant']):
    #   if not significant:
    #     ax.text(index[j] + i * bar_width - 0.1, 0.01, "*",
    #         fontdict = {'weight': 'bold', 'size': 16,
    #         'color': bar.patches[0].get_facecolor()})
        
    for j, significant in enumerate(vals['significant']):
      if not significant:
        ax.text(index[j] + i * bar_width - 0.1, 0.9, "*",
            fontdict = {'weight': 'bold', 'size': 16,
            'color': bar.patches[0].get_facecolor()})
      
    
  # print (plot_data)
  # set properties
  # (変更) 0.5に横線を引く
  ax.axhline(0.5, ls = "--", color = 'lightgray')
  # (変更) ターゲットクラス名表示
  target_class = results[0]['target_class'].title()
  ax.set_title('{} TCAV Scores (* is not rejected)'.format(target_class))
  ax.set_ylabel('TCAV Score')
  ax.set_ylim(0, 1)
  ax.set_xticks(index + num_bottlenecks * bar_width / 2)
  ax.set_xticklabels(plot_concepts)
  ax.legend(loc='upper left',bbox_to_anchor=(1.05, 1))
  fig.tight_layout()
  plt.show()


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
 
 
def get_dist(results):
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
 
 
def plot_concept_results(results, num_random_exp=100):
  # helper function, returns if this is a random concept
  def is_random_concept(concept):
    return 'random500_' in concept

  # print class, it will be the same for all
  # print("Class =", results[0]['target_class'])

  # prepare data
  # dict with keys of concepts containing dict with bottlenecks
  result_summary = {}
    
  for result in results:
    if result['cav_concept'] not in result_summary:
      result_summary[result['cav_concept']] = {}
    
    if result['bottleneck'] not in result_summary[result['cav_concept']]:
      result_summary[result['cav_concept']][result['bottleneck']] = []
    
    result_summary[result['cav_concept']][result['bottleneck']].append(result)

  # to plot, must massage data again 
  plot_data = {}
  plot_concepts = []
  
  num_concepts = 0
  
  
  # print concepts and classes with indentation
  for concept in result_summary:
        
    # if not random
    if not is_random_concept(concept):
      # print(" ", "Concept =", concept)
      plot_concepts.append(concept)
      num_concepts += 1

      for bottleneck in result_summary[concept]:
        i_ups = [item['i_up'] for item in result_summary[concept][bottleneck]]
                  
        if bottleneck not in plot_data:
          plot_data[bottleneck] = {'bn_vals': [], 'bn_stds': [], 'significant': []}
          
        plot_data[bottleneck]['bn_vals'].append(np.mean(i_ups))
        plot_data[bottleneck]['bn_stds'].append(np.std(i_ups))

  num_bottlenecks = len(plot_data)
  bar_width = 0.35
    
  # create location for each bar. scale by an appropriate factor to ensure 
  # the final plot doesn't have any parts overlapping
  index = np.arange(num_concepts) * bar_width * (num_bottlenecks + 1)
  
  
  
  
  # matplotlib
  fig, ax = plt.subplots()


  # draw all bottlenecks individually
  for i, [bn, vals] in enumerate(plot_data.items()):
    bar = ax.bar(index + i * bar_width, vals['bn_vals'],
        bar_width, yerr=vals['bn_stds'], label=bn)

  xlabel_name = ''
  if '-' in plot_concepts[0]:
    for w in plot_concepts[0].split('-')[:-1]:
      xlabel_name += w
    plot_concepts = [ c.split('-')[-1] for c in plot_concepts]
  if '_' in plot_concepts[0]:
    for w in plot_concepts[0].split('_')[:-1]:
      xlabel_name += w
    plot_concepts = [ c.split('_')[-1] for c in plot_concepts]
  xlabel_name += ' Concept'

  # print (plot_data)
  # set properties
  # (変更) 0.5に横線を引く
  ax.axhline(0.5, ls = "--", color = 'lightgray')
  # (変更) ターゲットクラス名表示
  target_class = results[0]['target_class'].title()
  ax.set_title('{} TCAV Scores'.format(target_class))
  ax.set_ylabel('TCAV Score')
  ax.set_xlabel(xlabel_name)
  ax.set_ylim(0, 1)
  ax.set_xticks(index + num_bottlenecks * bar_width / 2)
  #plt.xticks(fontsize=8)
  ax.set_xticklabels(plot_concepts)
  ax.legend(loc='upper left',bbox_to_anchor=(1.05, 1))
  fig.tight_layout()
  plt.show()
  
  

# sensitivityを得る
def get_sensitivity(results):
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
        
      random_i_ups[result['bottleneck']].append(result['val_directional_dirs_mean'])
    
  dist = {}
  
  # to plot, must massage data again 
  plot_data = {}
  plot_concepts = []

  # print concepts and classes with indentation
  num_concepts = 0
  for concept in result_summary:
    # if not random
    if not is_random_concept(concept):
      if concept not in dist:
        dist[concept] = {}
      plot_concepts.append(concept)      
      num_concepts += 1
      for bottleneck in result_summary[concept]:
        if bottleneck not in dist[concept]:
          dist[concept][bottleneck] = {}
        i_ups = [item['val_directional_dirs_mean'] for item in result_summary[concept][bottleneck]]
        dist[concept][bottleneck] = i_ups

        if bottleneck not in plot_data:
          plot_data[bottleneck] = {'bn_vals': [], 'bn_stds': [], 'significant': []}

        plot_data[bottleneck]['bn_vals'].append(np.mean(i_ups))
        plot_data[bottleneck]['bn_stds'].append(np.std(i_ups))
        
  plot_concepts += ['random']
  num_bottlenecks = len(plot_data)

  bar_width = 0.35
  index = np.arange(num_concepts) * bar_width * (num_bottlenecks + 1)
  fig, ax = plt.subplots()
  for i, [bn, vals] in enumerate(plot_data.items()):
    bar = ax.bar(index + i * bar_width, vals['bn_vals'],
        bar_width, yerr=vals['bn_stds'], label=bn)

  # (変更) ターゲットクラス名表示
  target_class = results[0]['target_class'].title()
  ax.set_title('{} Sensitivity'.format(target_class))
  ax.set_ylabel('Sensitivity')
  ax.set_xticks(index + num_bottlenecks * bar_width / 2)
  ax.set_xticklabels(plot_concepts)
  ax.legend(loc='upper left',bbox_to_anchor=(1.05, 1))
  fig.tight_layout()
  plt.show()

  # add random dist
  dist['random'] = {}
  for bottleneck in random_i_ups:
    dist['random'][bottleneck] = random_i_ups[bottleneck]
    
  # return dist