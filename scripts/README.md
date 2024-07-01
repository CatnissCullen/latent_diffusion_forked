# Note to Self

****



## 模型

**`LDM/ldm/models/diffusion/ddpm.py`**

***含 `class DDPM(pl.LightningModule)` 和 `class LatentDiffusion(DDPM)` ，LDM 类包含整个模型架构的 pipeline ：***

-   ***从输入图像感知压缩（实际使用 VAE ）&条件编码（实际使用 CLIP 中图像或文本编码器）***

-   ***到用于带条件学习噪声的特制 Unet*** 

    ***（在 `LDM/scripts/latent-diffusion/ldm/modules/diffusionmodules/openaimodel.py` ，使用 Cross-attention 进行条件注入，attention 模块在`LDM/scripts/latent-diffusion/ldm/modules/attention.py`）***  

-   ***使用 classifier-free 的条件引导方式计算预测噪声进行训练*** 

### VAE 

**加载预训练模型并冻结**

```python
def instantiate_first_stage(self, config):
        """
        pretrained VAE
        """
        model = instantiate_from_config(config)
        self.first_stage_model = model.eval()
        self.first_stage_model.train = disabled_train
        for param in self.first_stage_model.parameters():
            param.requires_grad = False
```

**用于感知压缩（模型对象已在调用处用输入图片进行初始化）**

```python
def get_first_stage_encoding(self, encoder_posterior):
        """
        Do Perceptual Compression on Input Image
        :param encoder_posterior: a VAE object initialized with Input Image
         e.g.   encoder_posterior = self.encode_first_stage(x)
                latent_vector = self.get_first_stage_encoding(encoder_posterior).detach()
        :return: Compressed Input
        """
        if isinstance(encoder_posterior, DiagonalGaussianDistribution):
            z = encoder_posterior.sample()
        elif isinstance(encoder_posterior, torch.Tensor):
            z = encoder_posterior
        else:
            raise NotImplementedError(f"encoder_posterior of type '{type(encoder_posterior)}' not yet implemented")
        return self.scale_factor * z
```

### CLIP 编码器

**加载预训练模型并冻结**

```python
def instantiate_cond_stage(self, config):
        """
        CLIP Encoders
        """
        if not self.cond_stage_trainable:
            if config == "__is_first_stage__":
                print("Using first stage also as cond stage.")
                self.cond_stage_model = self.first_stage_model
            elif config == "__is_unconditional__":
                print(f"Training {self.__class__.__name__} as an unconditional model.")
                self.cond_stage_model = None
                # self.be_unconditional = True
            else:
                model = instantiate_from_config(config)
                self.cond_stage_model = model.eval()
                self.cond_stage_model.train = disabled_train
                for param in self.cond_stage_model.parameters():
                    param.requires_grad = False
        else:
            assert config != '__is_first_stage__'
            assert config != '__is_unconditional__'
            model = instantiate_from_config(config)
            self.cond_stage_model = model
```

**用于条件编码**

```python
def get_learned_conditioning(self, c):
        """
        Apply Image-Encoder or Text-Encoder of CLIP to Condition
        :param c:
        :return: Encoded Condition
        """
        if self.cond_stage_forward is None:
            if hasattr(self.cond_stage_model, 'encode') and callable(self.cond_stage_model.encode):
                c = self.cond_stage_model.encode(c)
                if isinstance(c, DiagonalGaussianDistribution):
                    c = c.mode()
            else:
                c = self.cond_stage_model(c)
        else:
            assert hasattr(self.cond_stage_model, self.cond_stage_forward)
            c = getattr(self.cond_stage_model, self.cond_stage_forward)(c)
        return c
```



## 采样

**`LDM/ldm/models/diffusion/ddim.py`**

***只含 `class DDIMSampler(object)` ，用外部模型（ LDM 特制的 Unet ）进行初始化，而该 Sampler 只负责调用模型进行给定时间步 t 的 DDIM + Classifier-free 式噪声采样进而得样本采样*** 

```python
""" Classifier-free Sampling """
e_t_uncond, e_t = self.model.apply_model(x_in, t_in, c_in).chunk(2)
e_t = e_t_uncond + unconditional_guidance_scale * (e_t - e_t_uncond)
```

**完整采样过程**

用已训练的 LDM 对输入的分类标签条件进行采样

```python
with torch.no_grad():
    with model.ema_scope():
        uc = model.get_learned_conditioning(
            {model.cond_stage_key: torch.tensor(n_samples_per_class*[1000]).to(model.device)}
            )  # 空标签1000，即类别1000包含所有样本；用CLIP编码
        
        for class_label in classes:
            print(f"rendering {n_samples_per_class} examples of class '{class_label}' in {ddim_steps} steps and using s={scale:.2f}.")
            xc = torch.tensor(n_samples_per_class*[class_label])
            c = model.get_learned_conditioning(
                {model.cond_stage_key: xc.to(model.device)}
            )  # 实体标签；也用CLIP编码
            
            samples_ddim, _ = sampler.sample(S=ddim_steps,
                                             conditioning=c,
                                             batch_size=n_samples_per_class,
                                             shape=[3, 64, 64],
                                             verbose=False,
                                             unconditional_guidance_scale=scale,
                                             unconditional_conditioning=uc, 
                                             eta=ddim_eta)  # 采样

            x_samples_ddim = model.decode_first_stage(samples_ddim)  # 采样结果仍是感知压缩过的Latent，再用VAE超分才得到原大小图像
            x_samples_ddim = torch.clamp((x_samples_ddim+1.0)/2.0, 
                                         min=0.0, max=1.0)
            all_samples.append(x_samples_ddim)
```

