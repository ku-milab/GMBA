import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from math import log10
import random
import argparse
import datetime
from model import *
from utility import *


date_str = str(datetime.datetime.now().strftime('%Y%m%d.%H.%M.%S'))

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


torch.cuda.empty_cache()

Generator = Generator().cuda()
Discriminator = Discriminator().cuda()
MappingNetwork = MappingNetwork().cuda()

dex_age_classifier = DEXAgeClassifier().cuda()
for param in dex_age_classifier.parameters():
    param.requires_grad = False




"""
Load dataset
"""

train_npy = np.load('./train.npz', mmap_mode="r")
train_npy = train_npy['data']
train_csv = pd.read_csv('./train.csv')
val_npy = np.load('./val.npz', mmap_mode="r")
val_npy = val_npy['data']
val_csv = pd.read_csv('./val.csv')


train_npy = torch.from_numpy(train_npy)
train_npy = FF.rotate(train_npy, 270)
val_npy = torch.from_numpy(val_npy)
val_npy = FF.rotate(val_npy, 270)




"""
Test
"""
wrapped_dex_age_classifier = GradiendtClassifierWrapper()
wrapped_dex_age_classifier.cuda()


def main():


    START_AGE = 1
    END_AGE = 101

    ####### Loss functions

    Age_loss = DEX_L2loss(start_age=START_AGE, end_age=END_AGE).cuda()
    Id_loss = IDLoss().cuda()
    Cycle_loss = nn.L1Loss().cuda()

    ####### Optimizers
    optimizer_G = torch.optim.Adam(Generator.parameters(), lr=args.lr_gan)
    optimizer_M = torch.optim.Adam(MappingNetwork.parameters(), lr=args.lr_map)
    optimizer_D = torch.optim.Adam(Discriminator.parameters(), lr=args.lr_gan)

    ####### Training
    Generator.train()
    Discriminator.train()
    MappingNetwork.train()

    batch_size, epoch = args.batch_size, args.epoch

    total_val_idx2 = np.random.permutation(len(val_npy))

    for epoch in range(epoch+1):

        print('Epoch {}/{}'.format(epoch, args.epoch))

        D_loss = 0
        G_loss = 0
        MAE_final = 0

        total_idx = np.random.permutation(len(train_npy))
        total_idx2 = np.random.permutation(len(train_npy))


        train_loop = len(total_idx) - 59

        for idx in tqdm(range(0, train_loop, batch_size)):

            index1 = total_idx[idx:idx+batch_size]
            index2 = total_idx2[idx:idx+batch_size]
            # load dataset
            image1 = train_npy[index1]
            image2 = train_npy[index2]

            age1_ = torch.tensor(train_csv['age_label'].iloc[index1].values)
            age2_ = torch.tensor(train_csv['age_label'].iloc[index2].values)
            image1 = Variable(image1).cuda()
            image2 = Variable(image2).cuda()
            age1 = Variable(age1_).long().cuda()
            age2 = Variable(age2_).long().cuda()

            input_age = age_onehot(age1)
            input_age = input_age.cuda()

            target_age = age_onehot(age2)
            target_age = target_age.cuda()

            attr, mask = wrapped_dex_age_classifier(image1.float())

            # for making weights
            diff_age = (age2 - age1) / 100

            """
            Train Discriminator
            """

            optimizer_D.zero_grad()

            style = MappingNetwork(target_age)
            counter_map = Generator(image1.float(), style, attr, mask)
            fake_MRI = counter_map + image1.float()

            # Real Image
            pred_real = Discriminator(image2.float(), style.detach())

            # Fake Image
            pred_fake = Discriminator(fake_MRI.detach(), style.detach())

            # Discriminator Loss

            loss_D = 0.5 * (torch.mean((pred_real - 1) ** 2) + torch.mean(pred_fake ** 2))


            loss_D.backward()
            optimizer_D.step()


            """
            Train Generator
            """
            optimizer_G.zero_grad()
            optimizer_M.zero_grad()

            style = MappingNetwork(target_age)
            counter_map = Generator(image1.float(), style, attr, mask)
            fake_MRI = counter_map + image1.float()

            pred_real2 = Discriminator(fake_MRI, style.detach())
            loss_G_real = 0.5 * torch.mean((pred_real2 - 1) ** 2)

            ### Pretrained DEX age classifier
            final_pred_age = dex_age_classifier(fake_MRI.cuda())

            ## Age Prediction Loss
            loss_A_final = Age_loss(final_pred_age, age2)

            # calculate MAE
            mm = nn.Softmax(dim=1)
            output_softmax = mm(final_pred_age)
            aa = torch.arange(START_AGE, END_AGE + 1, dtype=torch.float32).cuda()
            mean = (output_softmax * aa).sum(1, keepdim=True).cpu().data.numpy()
            pred = np.around(mean)
            mae_final = np.absolute(pred.squeeze() - age2.cpu().data.numpy()).sum() / batch_size


            ## Identity Preservation Loss
            weights = compute_cosine_weights(diff_age)
            loss_id = Id_loss(fake_MRI.cuda(), image1.float(), weights)

            ## Cycle Consistency Loss
            style_input = MappingNetwork(input_age)
            counter_map_recon = Generator(fake_MRI.cuda(), style_input, attr, mask)
            fake_Recon = counter_map_recon + fake_MRI.cuda()

            loss_cycle = Cycle_loss(image1.float(), fake_Recon) # L1


            loss_G = loss_G_real + 0.05 * loss_A_final + 0.2 * loss_id + 0.5 * loss_cycle
            loss_G.backward()
            optimizer_G.step()
            optimizer_M.step()



            D_loss += loss_D.item()
            G_loss += loss_G_real.item()
            MAE_final += mae_final.item()



        """
        Quantitative Evaluation Metric
        # MAE
        """

        MAE_val = 0
        Val_A_loss = 0

        Generator.eval()
        MappingNetwork.eval()

        total_val_idx = np.arange(len(val_npy))

        val_loop = len(total_val_idx)

        for idx in tqdm(range(0, val_loop, batch_size)):

            index = total_val_idx[idx:idx + batch_size]
            index2 = total_val_idx2[idx:idx + batch_size]
            val_img_ = val_npy[index]
            val_age_ = torch.tensor(val_csv['age_label'].iloc[index].values)
            val_age2_ = torch.tensor(val_csv['age_label'].iloc[index2].values)

            val_img = Variable(val_img_).cuda()
            val_age2 = Variable(val_age2_).long().cuda()

            target_age = age_onehot(val_age2)
            target_age = target_age.cuda()

            attr, mask = wrapped_dex_age_classifier(val_img.float())


            with torch.no_grad():
                style = MappingNetwork(target_age)
                counter_map = Generator(val_img.float(), style, attr, mask)
                fake_MRI_val = counter_map + val_img.float()

                """
                MAE
                """
                ### Pretrained DEX age classifier
                final_pred_age = dex_age_classifier(fake_MRI_val.cuda())

                # calculate val loss
                val_loss = Age_loss(final_pred_age, val_age2)
                Val_A_loss += val_loss.item()

                # calculate MAE
                output_softmax = mm(final_pred_age)
                mean = (output_softmax * aa).sum(1, keepdim=True).cpu().data.numpy()
                pred = np.around(mean)
                mae_val = np.absolute(pred.squeeze() - val_age2.cpu().data.numpy()).sum() / batch_size
                MAE_val += mae_val

        print('D_loss:{:.3f} G_loss:{:.3f} Val_A_loss:{:.3f}'
              .format(D_loss / (train_loop / batch_size),
                      G_loss / (train_loop / batch_size),
                      Val_A_loss / (val_loop / batch_size)))

        print('MAE_final:{:.3f} MAE_val:{:.3f}'
              .format(MAE_final / (train_loop / batch_size),
                      MAE_val / (val_loop / batch_size)))


        Generator.train()
        MappingNetwork.train()


        if epoch % 5 == 0:

            save_dir = './' + 'batch_size' + str(
                args.batch_size) + 'lr_age_' + str(args.lr_age) + '_lr_gan_' + str(args.lr_gan) + '_scheduler_' + str(
                args.scheduler) + '_' + date_str + '/epoch_%d' % epoch

            if not os.path.exists(save_dir):
                os.makedirs(save_dir)


        if epoch > 99 and epoch % 10 == 0:
            torch.save(Generator.state_dict(), save_dir + '/Generator_epoch_%d.pth' % epoch)
            torch.save(MappingNetwork.state_dict(), save_dir + '/MappingNetwork_epoch_%d.pth' % epoch)

if __name__ == '__main__':
    main()
