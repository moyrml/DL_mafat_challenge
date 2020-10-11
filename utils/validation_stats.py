import numpy as np
import pandas as pd

def show_validation_accuracy(valid_x, valid_y, folds, model_folder, save_name, batch_size, model = None):
  device = 'cuda'

  submission =  pd.DataFrame()
  submission['segment_id'] = list(range(len(valid_y)))
  submission['prediction'] = 0
  
  confidence =  pd.DataFrame()
  confidence['segment_id'] = list(range(len(valid_y)))
  confidence['prediction'] = 0
  
  temp_set = IQ_data(valid_x, None)
  test_loader = DataLoader(temp_set, batch_size = batch_size)
  test_y = valid_y


  for fold in range(folds):
    test_pred = list()
    model.load_state_dict(torch.load(f'/content/gdrive/My Drive/mafat/{model_folder}/model_{save_name}_{1}.pth'))

    model.eval()

    with torch.no_grad():
      for i, batch in enumerate(test_loader):
        output = model(batch.to(device))
        test_pred.extend(
            torch.sigmoid(output.detach().cpu().view(-1)).tolist()
            )

    submission[f'fold {fold}'] = np.array(test_pred)
    temp = [int(np.round(x)) for x in test_pred]
    confidence[f'fold {fold}'] = np.array(temp)
    
  print("#######################################################################")
  submission['prediction'] = submission[[f'fold {i}' for i in range(folds)]].median(1).astype('float')

  print("With Median")
  temp = submission['prediction']
  temp2 = temp
  temp = np.round(temp)
  temp = [int(item) for item in temp]
  wrong, right = pred_description(temp, test_y)
  print("#######################################################################\n\n")

  confidence['most'] = confidence[[f'fold {i}' for i in range(folds)]].mode(1).astype('int')
  return wrong, right, confidence
 


def show_validation_accuracy_two(valid_x, valid_y, fold, model_folder, save_name, model, batch_size, mode = 0, transforms=None):
  device = 'cuda'
  wrong_indices = []
  right_indices = []

  temp_set = IQ_data(valid_x, None, transforms=transforms)
  test_loader = DataLoader(temp_set, batch_size = batch_size)
  test_y = valid_y

  test_pred = list()
  model.load_state_dict(torch.load(f'/content/gdrive/My Drive/mafat/{model_folder}/model_{save_name}_{fold}.pth'))

  model.eval()

  with torch.no_grad():
    for i, batch in enumerate(test_loader):
      output = model(batch.to(device))
      test_pred.extend(
          torch.sigmoid(output.detach().cpu().view(-1)).tolist()
          )

  pred = np.array(test_pred) 
  pred1 = np.round(pred)
  pred1 = [int(item) for item in pred1]
  if mode == 1:
    print("#######################################################################")
    temp_wrong_indices, temp_right_indices = pred_description(pred1, test_y)
    print("#######################################################################")
  return pred



def more_pred_info(probabilities, actual_labels, temp_wrong_indices, temp_right_indices):
    one_indices = np.where(actual_labels == 1)
    one_indices = one_indices[0]
    one_correct_indices = np.intersect1d(temp_right_indices, one_indices)
    one_incorrect_indices = np.intersect1d(temp_wrong_indices, one_indices)
    # print(one_indices)
    # print(temp_right_indices)
    # print(one_correct_indices)
    probabilities = np.array(probabilities)
    ones_correct_pred = probabilities[one_correct_indices]
    ones_wrong_pred = probabilities[one_incorrect_indices]
    
    print(f"Average Probability for Correct 1's: {np.mean(ones_correct_pred)}")
    print(f"STD for Correct 1's: {np.std(ones_correct_pred)}")
    print(f"Average Probability for Wrong 1's: {np.mean(ones_wrong_pred)}")
    print(f"STD for Wrong 1's: {np.std(ones_wrong_pred)}")
    
    zero_indices = np.where(actual_labels == 0)
    zero_indices = zero_indices[0]
    zero_correct_indices = np.intersect1d(temp_right_indices, zero_indices)
    zero_incorrect_indices = np.intersect1d(temp_wrong_indices, zero_indices)
    
    zeros_correct_pred = probabilities[zero_correct_indices]
    zeros_wrong_pred = probabilities[zero_incorrect_indices]
    
    print(f"Average Probability for Correct 0's: {np.mean(zeros_correct_pred)}")
    print(f"STD for Correct 0's: {np.std(zeros_correct_pred)}")
    print(f"Average Probability for Wrong 0's: {np.mean(zeros_wrong_pred)}")
    print(f"STD for Wrong 0's: {np.std(zeros_wrong_pred)}")
    
    upper_conf_lim = 0.65
    lower_conf_lim = 0.27
    
    items_to_keep = np.where(probabilities > upper_conf_lim)
    items_to_keep = items_to_keep[0]
    items_to_keep_two = np.where(probabilities < lower_conf_lim)
    items_to_keep_two = items_to_keep_two[0]

    good_indices = np.concatenate((np.array(items_to_keep), np.array(items_to_keep_two)))
    new_probabilities = probabilities[list(good_indices)]
    new_actuals = actual_labels[list(good_indices)]
    
    print("Cleaned Prediction Vectors")
    pred_description(list(new_probabilities), list(new_actuals))
    
    items_to_remove = np.where(probabilities <= upper_conf_lim)
    items_to_remove = items_to_remove[0]
    items_to_remove_two = np.where(probabilities >= lower_conf_lim)
    items_to_remove_two = items_to_remove_two[0]

    bad_indices = np.intersect1d(np.array(items_to_remove), np.array(items_to_remove_two))
    
    return bad_indices
    
    

def decision(probability):
    return random.random() < probability
    

 

def pred_description(predictions, actuals, produce_roc_stats = False):
  total_items = len(predictions)
  print(f"Number of Total Items: {len(actuals)}")
    
  if produce_roc_stats:
      fpr, tpr, _ = roc_curve(actuals, predictions)
      roc_auc = auc(fpr, tpr)
      
      return dict(
          FPR = fpr,
          TPR = tpr,
          ROC_AUC = roc_auc
          )
          
  false_pos = 0
  false_neg = 0
  true_pos = 0
  true_neg = 0

  for i, item in enumerate(predictions):
    item = round(item)
    item2 = round(actuals[i])

    if item2 == 0:
      if item == 0:
        true_neg += 1
      if item == 1:
        false_pos += 1

    if item2 == 1:
      if item == 0:
        false_neg += 1
      if item == 1:
        true_pos += 1
        
  print(f"False when actually true: {false_neg}")
  print(f"True when actually false: {false_pos}")
  print(f"True when actually true: {true_pos}")
  print(f"False when actually false: {true_neg}")

  correct_count = 0
  total_count = 0
  for item1, item2 in list(zip(list(predictions), list(actuals))):
    item1 = int(np.round(item1))
    if item1 == item2:
      correct_count += 1
    total_count += 1

  indices_where_wrong = np.where(predictions != actuals)
  indices_where_wrong = indices_where_wrong[0]
  
  indices_where_right = np.where(predictions == actuals)
  indices_where_right = indices_where_right[0]
  
  print(f"Final Accuracy: {correct_count / total_count * 100}")
  
  return None



def describe_targets(actuals):
  total_items = len(actuals)  
  min_item_count = np.inf
  # print(f"Number of Total Items: {len(actuals)}")
  for item in np.unique(actuals):
    item_count = np.sum(np.where(actuals == item, 1, 0))
    print(f"Actual {int(item)} Count: {np.sum(np.where(actuals == item, 1, 0))}")
    # print(f"Actual {item} Percentage: {(np.sum(np.where(actuals == item, 1, 0))/total_items)*100}")
    if item_count < min_item_count:
      min_item = item
      min_item_count = item_count

  return min_item_count
  
  
