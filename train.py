import torch
import torch.nn as nn


def train_loop(model_folder, save_file, fold, epochs, model, train_loader, valid_loader, criterion, optimizer, lr_scheduler, output_progress = True, criterion_two = None):

  device = "cuda"
  training_loss = list()
  validation_losses = list()

  val_preds = list()
  val_targets = list()

  train_preds = list()
  train_targets = list()

  best_val_loss = np.inf
  

  for epoch in range(epochs):
    if output_progress:
      print(f"In Epoch {epoch+1} out of {epochs}.")


    for param_group in optimizer.param_groups:
      cur_lr = param_group['lr']
    
    if criterion_two == None or cur_lr > 0.0001:
      print("Using Focal Loss......")
      mode = 1
    else:
      print("Using Secondary Loss....")
      mode = 2
      
    suffix = ''
    model.train(True)
    epoch_loss = 0
    val_preds = list()
    val_targets = list()

    if output_progress:
      out = display(progress(0, len(train_loader)), display_id=True)
    
    for i, batch in enumerate(train_loader):
      if output_progress:
        out.update(progress(i, len(train_loader)))
      
    #   output = model(batch[0].to(device))
      
      if mode == 1:
        output = model(batch[0].to(device))
        loss = criterion(output, batch[1].to(device).view(-1,1).float())
        
      elif mode == 2:
        # do something else
        continue
    
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
      epoch_loss += loss.item() / len(train_loader) 
      
      if epoch == epochs-1:                                             # if last epoch
        train_preds.extend(
          torch.sigmoid(output.detach().cpu().view(-1)).tolist()        # extending train predictions
          )
        
        train_targets.extend(                                           # extending train actuals
            batch[1].tolist()
        )

    training_loss.append(epoch_loss)                                    # training loss append each epoch

    epoch_loss = 0        # reset epoch loss
    
    model.eval()          # evaluation mode
    with torch.no_grad():
      for i,batch in enumerate(valid_loader):
        output = model(batch[0].to(device))
        loss = criterion(output, batch[1].to(device).view(-1,1).float())
        epoch_loss += loss.item() / len(valid_loader)

        if epoch == epochs-1 or output_progress:
          val_preds.extend(
            torch.sigmoid(output.detach().cpu().view(-1)).tolist()
            )
          
          val_targets.extend(
              batch[1].tolist()
          )
    
    if output_progress:
        print(val_targets)
        print(val_preds)
        pred_description(val_preds, val_targets)
    
    lr_scheduler.step(epoch_loss)             # each epoch
    validation_losses.append(epoch_loss)      # each epoch


    if epoch_loss < best_val_loss:
      notebook_root_two = 'My Drive/mafat'
      torch.save(model.state_dict(), f'/content/gdrive/My Drive/mafat/{model_folder}/model_{save_file}_{fold}.pth')
      suffix = 'Saved best model.'
      best_val_loss = epoch_loss

    print(f'Fold: {fold}, Epoch {epoch + 1}\tTrain Loss {training_loss[-1]:0.4f}\tValidation Loss {validation_losses[-1]:0.4f}\t {suffix}')

  return training_loss, validation_losses, val_targets, val_preds, train_targets, train_preds



def train_loop_two(model_folder, save_file, fold, epochs, model, train_loader, valid_loader, criterion, criterion_two, optimizer, lr_scheduler, prob_switch = 0.75):

  device = "cuda"

  training_loss = list()
  validation_losses = list()

  val_preds = list()
  val_targets = list()

  train_preds = list()
  train_targets = list()

  best_val_loss = np.inf
  

  for epoch in range(epochs):
    # if output_progress:
    #   print(f"In Epoch {epoch+1} out of {epochs}.")

    suffix = ''
    model.train(True)
    epoch_loss = 0

    standard_loss = decision(prob_switch)
    # prob_switch = prob_switch - epoch_decay

    for param_group in optimizer.param_groups:
      cur_lr = param_group['lr']
    print(cur_lr)

    if standard_loss or cur_lr > 0.0001:
      print("Using Focal Loss......")
    else:
      print("Using Triplet Loss....")

    # if output_progress:
    #   out = display(progress(0, len(train_loader)), display_id=True)
    
    for i, batch in enumerate(train_loader):

      optimizer.zero_grad()

      # anchor_out = model(batch[0].to(device))
      # positive_out = model(batch[1].to(device))
      # negative_out = model(batch[2].to(device))

      if standard_loss:
        anchor_out = model(batch[0].to(device))
        loss = criterion_two(anchor_out, batch[3].to(device).view(-1,1).float())
        # switch = False
      else:
        anchor_out = model(batch[0].to(device))
        positive_out = model(batch[1].to(device))
        negative_out = model(batch[2].to(device))
        loss = criterion(anchor_out, positive_out, negative_out, batch[3].to(device).view(-1,1).float())
        # switch = True
        

      # optimizer.zero_grad()
      # loss = criterion(anchor_out, positive_out, negative_out)
 
      loss.backward()
      optimizer.step()

      epoch_loss += loss.item() / len(train_loader) 
      
      if epoch == epochs-1:                                             # if last epoch
        train_preds.extend(
          # torch.sigmoid(output.detach().cpu().view(-1)).tolist()        # extending train predictions
            torch.sigmoid(anchor_out.detach().cpu().view(-1)).tolist()

          )
        
        train_targets.extend(                                           # extending train actuals
            batch[1].tolist()
        )

    training_loss.append(epoch_loss)                                    # training loss append each epoch


    epoch_loss = 0        # reset epoch loss
    
    model.eval()          # evaluation mode
    with torch.no_grad():
      for i,batch in enumerate(valid_loader):
        
        anchor_out = model(batch[0].to(device))
        # positive_out = model(batch[1].to(device))
        # negative_out = model(batch[2].to(device))        
        
        loss = criterion_two(anchor_out, batch[3].to(device).view(-1,1).float())
        # loss = criterion(anchor_out, positive_out, negative_out)

        epoch_loss += loss.item() / len(valid_loader)

        if epoch == epochs-1:
          val_preds.extend(
            # torch.sigmoid(output.detach().cpu().view(-1)).tolist()
            torch.sigmoid(anchor_out.detach().cpu().view(-1)).tolist()

            )
          
          val_targets.extend(
              batch[3].tolist()
          )

    lr_scheduler.step(epoch_loss)             # each epoch
    validation_losses.append(epoch_loss)      # each epoch


    if epoch_loss < best_val_loss:
      notebook_root_two = 'My Drive/mafat'
      torch.save(model.state_dict(), f'/content/gdrive/My Drive/mafat/{model_folder}/model_{save_file}_{fold}.pth')
      suffix = 'Saved best model.'
      best_val_loss = epoch_loss

    print(f'Fold: {fold}, Epoch {epoch + 1}\tTrain Loss {training_loss[-1]:0.4f}\tValidation Loss {validation_losses[-1]:0.4f}\t {suffix}')

  return training_loss, validation_losses, val_targets, val_preds, train_targets, train_preds

 
