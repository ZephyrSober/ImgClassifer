import mindspore as ms 
from mindspore import nn
import os

def train_one_epoch(model,dataset,loss_fn,optimizer,epoch,total_epochs):
    model.set_train(True)
   
    def forward_fn(data,label):
        logits=model(data)
        loss=loss_fn(logits,label)
        return loss
    
    grad_fn=ms.value_and_grad(forward_fn,None,optimizer.parameters)
    total_loss=0.0
    steps=dataset.get_dataset_size()
    for batch_idx,(data,label) in enumerate(dataset.create_tuple_iterator()):
        loss,grads=grad_fn(data,label)
        optimizer(grads)
        total_loss+=loss.asnumpy()

        if(batch_idx+1)%10==0 or (batch_idx+1)==steps:
            print(f"Epoch [{epoch}/{total_epochs}], Step [{batch_idx+1}/{steps}], Loss: {loss.asnumpy():.4f}")

    avg_loss=total_loss/steps
    return avg_loss

def validate_one_epoch(model,dataset,loss_fn):
    model.set_train(False)
    total_loss=0.0
    correct_preds=0
    total_samples=0
    steps=dataset.get_dataset_size()
    for data,label in dataset.create_tuple_iteretor():
        logits=model(data)
        loss=loss_fn(logits,label)
        total_loss+=loss.asnumpy()

        preds=logits.argmax(axis=1)
        correct_preds+=(preds==label).sum().asnumpy()
        total_samples+=label.shape[0]

    avg_loss=total_loss/steps
    accuracy=correct_preds/total_samples
    print(f"Validation Results - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")
    return avg_loss, accuracy

def test_one_epoch(model,dataset,loss_fn):
    print("="*30)
    print("--- 启动最终测试集评估 ---")
    avg_loss, accuracy = validate_one_epoch(model, dataset, loss_fn)
    print(f"Final Test Results - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")
    print("="*30)
    return avg_loss, accuracy

def save_checkpoint_if_best(model, current_acc, best_acc, save_dir="./checkpoints", model_name="best_model.ckpt"):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    if current_acc > best_acc:
        print(f"Accuracy improved from {best_acc:.4f} to {current_acc:.4f}. Saving new best checkpoint...")
        save_path = os.path.join(save_dir, model_name)
        ms.save_checkpoint(model, save_path)
        return current_acc
    return best_acc