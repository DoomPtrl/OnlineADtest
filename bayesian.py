from scipy.optimize import minimize
import numpy as np
from scipy.stats import norm

def likelihoods4censored_data(params, censored_data, min_score=None, max_score=None):
    mean, std = params[0], params[1]
    if min_score is None:
        min_score = censored_data.min()
    if max_score is None:
        max_score = censored_data.max()
    likelihoods = norm.cdf(censored_data+1, loc=mean, scale=std)-norm.cdf(censored_data, loc=mean, scale=std)
    likelihoods[censored_data <= min_score] = norm.cdf(min_score+1, loc=mean, scale=std)
    likelihoods[censored_data >= max_score] = 1 - norm.cdf(max_score, loc=mean, scale=std)
    # print('mean', mean, 'std', std)
    # print(censored_data[:10])
    # print(likelihoods[:10])
    return likelihoods

def neg_log_likelihood4censored_data(params, censored_data, min_score=None, max_score=None):
    if min_score is None:
        min_score = censored_data.min()
    if max_score is None:
        max_score = censored_data.max()
    likelihoods=likelihoods4censored_data(params, censored_data, min_score, max_score)
    return -np.sum(np.log(np.clip(likelihoods, 1e-10, None)))  # Avoid log of 0 with clipping

def parameters4censored_data(censored_data, min_score=None, max_score=None):
    if max_score is None:
        max_score = censored_data.max()
    if min_score is None:
        min_score = censored_data.min()
    initial_params = [np.mean(censored_data), np.std(censored_data)]
    result = minimize(
        neg_log_likelihood4censored_data, 
        initial_params, 
        args=(censored_data, min_score, max_score),
        bounds=[(None, None), (1e-10, None)]
    )
    corrected_params= [result.x[0], result.x[1]]
    return corrected_params

def parameters4each_class(data,min_score=None, max_score=None):
    if max_score is None:
        max_score = data['SCORE'].max()
    if min_score is None:
        min_score = data['SCORE'].min()
    corrected_params = {}
    for diagnosis, group_data in data.groupby('DIAGNOSIS')['SCORE']:
        # print('diagnosis', diagnosis)
        # print('group_data', group_data)
        corrected_params[diagnosis] = parameters4censored_data(group_data, min_score, max_score)
    # print(corrected_params)
    return corrected_params

def classifier_loss(weights, data, criteria, criteria_range):
    data['SCORE'] = data[criteria].dot(weights)
    max_score = weights.sum()*criteria_range
    params=parameters4censored_data(data['SCORE'], max_score=max_score)
    log_score_priors = -neg_log_likelihood4censored_data(params, data['SCORE'], 0, max_score)
    log_score_class =0
    params=parameters4each_class(data,max_score=max_score)
    for diagnosis, group_data in data.groupby('DIAGNOSIS')['SCORE']:
        log_score_class -= neg_log_likelihood4censored_data(params[diagnosis], group_data,min_score=0,max_score=max_score)
    total_loss = -(log_score_class-log_score_priors)
    print(weights, total_loss)
    return total_loss

def posterior_probability(data, max_score, class_priors=None):
    if class_priors is None:
        class_priors = data['DIAGNOSIS'].value_counts(normalize=True)
        
    parameters = parameters4each_class(data, max_score=max_score)
    likelihoods_list = []
    # print(parameters)
    # Compute likelihoods for each class
    for diagnosis in sorted(data['DIAGNOSIS'].unique()):
        params = parameters[diagnosis]        
        # Compute likelihoods for all samples under this class
        # print(data['SCORE'].values[:10])
        likelihood = likelihoods4censored_data(params, data['SCORE'].values,min_score=0, max_score=max_score)
        # print('likelihood', likelihood[:10])
        likelihood*=class_priors[diagnosis]
        likelihoods_list.append(likelihood)
    likelihoods_array = np.stack(likelihoods_list, axis=0).T  # Shape: (num_samples, num_classes)    
    likelihoods_array /= likelihoods_array.sum(axis=1, keepdims=True)    
    return likelihoods_array

def classifier_loss2(weights, data, criteria, criteria_range, ground_truth, class_priors):
    data['SCORE'] = data[criteria].dot(weights)
    max_score = weights.sum() * criteria_range
    posterior_probs = posterior_probability(data, max_score, class_priors)
    total_loss = -np.sum(ground_truth * np.log(np.clip(posterior_probs, 1e-10, None)))
    print(weights, total_loss)
    return total_loss

def bayesian_classifier(data, max_score, class_priors=None):
    if class_priors is None:
        class_priors = data['DIAGNOSIS'].value_counts(normalize=True)
    return np.argmax(posterior_probability(data, max_score, class_priors), axis=1)
    
