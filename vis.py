import cv2
import os
import numpy as np
import torch
from torchvision import transforms
from PIL import Image
import resnet12
import datasets
from args import args


# 图片预处理
def img_preprocess(img_path):
    norm = transforms.Normalize(np.array([x / 255.0 for x in [125.3, 123.0, 113.9]]),
                                np.array([x / 255.0 for x in [63.0, 62.1, 66.7]]))
    all_transforms =torch.nn.Sequential(transforms.Resize(92), transforms.CenterCrop(84), norm)
    all_transforms_ = torch.nn.Sequential(transforms.Resize(4*92), transforms.CenterCrop(4*84))
    img=all_transforms(transforms.ToTensor()(np.array(Image.open(img_path).convert('RGB'))))
    img_ori = torch.from_numpy(np.array(Image.open(img_path)).transpose((2, 0, 1))).contiguous()
    img_ori=all_transforms_(img_ori).transpose(0,2).transpose(0,1)
    img = img.unsqueeze(0)					# 3
    img_ori=cv2.cvtColor(np.asarray(img_ori), cv2.COLOR_RGB2BGR)


    return img,img_ori

# 定义获取梯度的函数
def backward_hook(module, grad_in, grad_out):
    grad_block.append(grad_out[0].detach())

# 定义获取特征图的函数
def farward_hook(module, input, output):
    fmap_block.append(output)

# 计算grad-cam并可视化
def cam_show_img(img, feature_map, grads, out_dir):
    H, W, _ = img.shape
    cam = np.zeros(feature_map.shape[1:], dtype=np.float32)		# 4
    grads = grads.reshape([grads.shape[0],-1])					# 5
    weights = np.mean(grads, axis=1)							# 6
    for i, w in enumerate(weights):
        cam += w * feature_map[i, :, :]							# 7
    cam = np.maximum(cam, 0)
    cam = cam / cam.max()
    cam = cv2.resize(cam, (W, H))

    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    print(heatmap.shape,img.shape)
    cam_img = 0.5 * heatmap + 0.5 * img

    cam_img=cv2.resize(cam_img,(640,640))
    import os
    num=len(os.listdir(out_dir))
    path_cam_img = os.path.join(out_dir, "cam"+str(num)+".jpg")
    cv2.imwrite(path_cam_img, cam_img)

if __name__ == '__main__':
    path_img = './cam/20332_00000791.jpg'
    output_dir = './cam'


    # 存放梯度和特征图
    fmap_block = list()
    grad_block = list()

    # 图片读取；网络加载
    img_input,img_ori = img_preprocess(path_img)

    loaders, input_shape, num_classes, few_shot, top_5 = datasets.get_dataset(args.dataset)
    print(input_shape, num_classes, few_shot, top_5 )
    num_classes, val_classes, novel_classes, elements_per_class = num_classes
    model=resnet12.ResNet12(64, input_shape, num_classes, few_shot, args.rotations).to(args.device)

    # 加载预训练模型
    pthfile = './checkpoint/QFSD1.pt11'
    model.load_state_dict(torch.load(pthfile, map_location=torch.device(args.device)))
    model.eval()														# 8
    print(model)

    # 注册hook
    model.layers[-1].register_forward_hook(farward_hook)	# 9
    model.layers[-1].register_backward_hook(backward_hook)

    # forward
    print(model(img_input.to(args.device)))
    output, features = model(img_input.to(args.device))
    idx = np.argmax(output.cpu().data.numpy())
    print("predict: {}".format(idx))

    # backward
    model.zero_grad()
    class_loss=features[0,24]
    #class_loss = output[0,idx]
    class_loss.backward()

    # 生成cam
    grads_val = grad_block[0].cpu().data.numpy().squeeze()
    fmap = fmap_block[0].cpu().data.numpy().squeeze()

    # 保存cam图片
    cam_show_img(img_ori, fmap, grads_val, output_dir)


