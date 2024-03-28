import os
import torch
import torch.nn as nn
import numpy as np

from sssd.utils.util import (
    find_max_epoch, print_size,
    training_loss, sampling,
    calc_diffusion_hyperparams
    )

from sssd.utils.util import TS_Dataset
from torch.utils.data import DataLoader

from sssd.imputers import SSSDS4Imputer

from tqdm import tqdm



class SSSDS4():
    def __init__(
            self,
            ts_len=0,
            ts_dim=0,
            T=200,
            beta_0=0.0001,
            beta_T=0.02,
            num_res_layers=36,
            res_channels=256, 
            skip_channels=256,
            diffusion_step_embed_dim_in=128,
            diffusion_step_embed_dim_mid=512,
            diffusion_step_embed_dim_out=512,
            s4_d_state=64,
            s4_dropout=0.0,
            s4_bidirectional=1,
            s4_layernorm=1
    ):
        
        # save diffustion config
        self._set_diffusion_config(
            T, beta_0, beta_T
        )

        # save model config
        self._set_model_config(
            ts_dim, ts_dim,
            num_res_layers,
            res_channels, 
            skip_channels,
            diffusion_step_embed_dim_in,
            diffusion_step_embed_dim_mid,
            diffusion_step_embed_dim_out,
            ts_len,
            s4_d_state,
            s4_dropout,
            s4_bidirectional,
            s4_layernorm
        )

        # save params
        self.params={
            "diffusion_config": self.diffusion_config,
            "model_config": self.model_config
        }

        self.ckpt_epoch = 0


    def _set_diffusion_config(self, T, beta_0, beta_T):
        self.diffusion_config={
            "T": T,
            "beta_0": beta_0,
            "beta_T": beta_T
        }
        

    def _set_model_config(
            self,
            in_channels,
            out_channels,
            num_res_layers,
            res_channels, 
            skip_channels,
            diffusion_step_embed_dim_in,
            diffusion_step_embed_dim_mid,
            diffusion_step_embed_dim_out,
            s4_lmax,
            s4_d_state,
            s4_dropout,
            s4_bidirectional,
            s4_layernorm
    ):
        self.model_config={
            "in_channels": in_channels, 
            "out_channels": out_channels,
            "num_res_layers": num_res_layers,
            "res_channels": res_channels, 
            "skip_channels": skip_channels,
            "diffusion_step_embed_dim_in": diffusion_step_embed_dim_in,
            "diffusion_step_embed_dim_mid": diffusion_step_embed_dim_mid,
            "diffusion_step_embed_dim_out": diffusion_step_embed_dim_out,
            "s4_lmax": s4_lmax,
            "s4_d_state":s4_d_state,
            "s4_dropout":s4_dropout,
            "s4_bidirectional":s4_bidirectional,
            "s4_layernorm":s4_layernorm
        }

    
    def set_train_config(
                self,
                output_directory,
                epochs=200,
                epochs_per_ckpt=50,
                epochs_per_val=10,
                learning_rate=2e-4,
                only_generate_missing=True,
                missing_mode="rm",
                missing_r="rand",
                batch_size=8,
                verbose=True,
        ):
            self.train_config={
                "epochs": epochs,
                "epochs_per_ckpt": epochs_per_ckpt,
                "epochs_per_val": epochs_per_val,
                "learning_rate": learning_rate,
                "only_generate_missing": only_generate_missing,
                "missing_mode": missing_mode,
                "missing_r": missing_r,
                "batch_size": batch_size,
                "verbose": verbose
            }
            self.params["train_config"] = self.train_config
            self.output_directory = output_directory
            if not os.path.isdir(self.output_directory):
                os.makedirs(self.output_directory)
                os.chmod(self.output_directory, 0o775)
            
            if self.ckpt_epoch:
                return
            
            # predefine model
            self.net = SSSDS4Imputer(**self.model_config).cuda()
            self.optimizer = torch.optim.Adam(
                self.net.parameters(),
                lr=self.train_config["learning_rate"]
            )
            self.loss = nn.MSELoss()

            # calculate diffusion hyperparams
            self.diffusion_hyperparams = calc_diffusion_hyperparams(
                **self.diffusion_config)
            
            # map diffusion hyperparameters to gpu
            for key in self.diffusion_hyperparams:
                if key != "T":
                    self.diffusion_hyperparams[key] = self.diffusion_hyperparams[key].cuda()
    

    def load_state(
            self,
            output_directory,
            ckpt_epoch='max'
        ):

            self.output_directory = output_directory
            if not os.path.isdir(self.output_directory):
                os.makedirs(self.output_directory)
                os.chmod(self.output_directory, 0o775)
            
            # load checkpoint if any
            if ckpt_epoch == 'max':
                ckpt_epoch = find_max_epoch(self.output_directory)
            if ckpt_epoch > 0:
                try:
                    
                    # load checkpoint file
                    model_path = os.path.join(self.output_directory, '{}.pkl'.format(ckpt_epoch))
                    checkpoint = torch.load(model_path, map_location='cpu')

                    # load configs
                    self._set_diffusion_config(**checkpoint["diffusion_config"])
                    self._set_model_config(**checkpoint["model_config"])
                    self.set_train_config(output_directory=self.output_directory,
                                          **checkpoint["train_config"])
                    self.ckpt_epoch = ckpt_epoch
                    # load model state and optimizer state
                    self.net.load_state_dict(checkpoint['model_state_dict'])
                    self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

                    print('Successfully loaded model at epoch {}.'.format(ckpt_epoch))

                except:
                    self.ckpt_epoch = 0
                    print('No valid checkpoint model found.')
            else:
                self.ckpt_epoch = 0
                print('No valid checkpoint model found.')


    def train(self, train, val):
        
        if self.train_config["verbose"]:
            print_size(self.net)
        print("Output directory: ", self.output_directory, flush=True)
        
        val, val_ori = val

        train = train.transpose(0, 2, 1)
        val = val.transpose(0, 2, 1)
        val_ori = val_ori.transpose(0, 2, 1)

        # train data loading
        train_ds = TS_Dataset(
            data=np.where(np.isnan(train), 0, train),
            obs_mask=~np.isnan(train),
            dataset_mode="Training",
            missing_mode=self.train_config["missing_mode"],
            missing_r=self.train_config["missing_r"]
        )
        train_loader = DataLoader(
            train_ds, shuffle=True,
            batch_size=self.train_config["batch_size"]
        )


        # val data loading
        val_ds = TS_Dataset(
            data=(np.where(np.isnan(val), 0, val), np.where(np.isnan(val_ori), 0, val_ori)),
            obs_mask=(~np.isnan(val), ~np.isnan(val_ori)),
            dataset_mode="Validation",
            missing_mode=self.train_config["missing_mode"],
            missing_r=0.5
        )
        val_loader = DataLoader(
            val_ds, shuffle=True,
            batch_size=min(val.shape[0], self.train_config["batch_size"])
        )
        print('Data loaded')
        





        # training
        for epoch in tqdm(range(self.ckpt_epoch, self.train_config["epochs"]),
                          total = self.train_config["epochs"],
                          initial = self.ckpt_epoch,
                          desc="Training the network",
                          disable = not self.train_config["verbose"]):

            # train step
            train_loss = self._step(train_loader, epoch, "Training")

            # validation step and output
            if (epoch + 1) % self.train_config["epochs_per_val"] == 0 or epoch == 0:
                val_loss = self._step(val_loader, epoch, "Validation")
                
                string =  "Epoch: {}/{}.. Training Loss: {:.2e}.. Validation Loss: {:.2e}.."
                string = string.format(epoch+1, self.train_config["epochs"], train_loss, val_loss)
                if self.train_config["verbose"]:
                    tqdm.write(string)
                else:
                    print(string)
            else:
                string =  "Epoch: {}/{}.. Training Loss: {:.2e}.."
                string = string.format(epoch+1, self.train_config["epochs"], train_loss)
                if self.train_config["verbose"]:
                    tqdm.write(string)
                else:
                    print(string)
                
            # save checkpoint
            if (epoch + 1) % self.train_config["epochs_per_ckpt"] == 0:

                checkpoint_name = '{}.pkl'.format(epoch + 1)
                torch.save(
                    {
                        "model_state_dict": self.net.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "diffusion_config": self.diffusion_config,
                        "model_config": self.model_config,
                        "train_config": self.train_config
                    },
                    os.path.join(self.output_directory, checkpoint_name)
                )
                if self.train_config["verbose"]:
                    tqdm.write('Model at epoch %s is saved' % (epoch+1))
                else:
                    print('Model at epoch %s is saved' % (epoch+1))
                


    def _step(self, data_loader, epoch, phase):

        if phase == "Training":
            self.net.train()
        else:
            self.net.eval()

        avg_loss = 0.0
        loss_masked = 0
        for batch, obs_mask, cond_mask in tqdm(data_loader, total=len(data_loader), leave=False,
                                 desc="Epoch {}, {} phase".format(epoch, phase),
                                 disable = not self.train_config["verbose"]):
            
            
            obs_mask = obs_mask.cuda()
            cond_mask = cond_mask.cuda()
            loss_mask = torch.logical_xor(cond_mask.bool(), obs_mask.bool())
            loss_mask = loss_mask.cuda()
            loss_masked += loss_mask.sum().item()
            
            # training
            if phase == "Training":
                batch = batch.cuda()

                self.optimizer.zero_grad()
                X = batch, batch, cond_mask, loss_mask
                loss = training_loss(self.net, self.loss, X, self.diffusion_hyperparams)

                loss.backward()
                self.optimizer.step()

            # validation
            else:
                batch, batch_ori = batch[0].cuda(), batch[1].cuda()
                with torch.no_grad():
                    imputed = sampling(
                        self.net,
                        self.diffusion_hyperparams,
                        cond = batch,
                        mask = cond_mask
                    )
                    
                    loss = self.loss(imputed[loss_mask], batch_ori[loss_mask])

            avg_loss += loss.item() * loss_mask.sum().item()

        avg_loss=float(avg_loss)
        return avg_loss / loss_masked
    

    def predict(self, data, batch_size):
        data = data.transpose(0, 2, 1)

        # create data loader
        dataset = TS_Dataset(
            data=data,
            obs_mask= ~np.isnan(data),
            dataset_mode="Inference"
        )
        data_loader = DataLoader(
            dataset, shuffle=False,
            batch_size = batch_size
        )

        results = []

        # sampling
        for batch, obs_mask in tqdm(
            data_loader, total=len(data_loader),
            leave=False, desc="Prediction",
            disable = not self.train_config["verbose"]
        ):
            batch = batch.cuda()
            obs_mask = obs_mask.cuda()
            imputed = sampling(
                        self.net,
                        self.diffusion_hyperparams,
                        cond = torch.where(obs_mask.bool(), batch, 0),
                        mask = obs_mask.float()
                    )
            results.append(imputed.cpu().numpy())
        
        return (np.concatenate(results)).transpose(0, 2, 1)
                    

