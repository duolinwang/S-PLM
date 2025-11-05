## S-PLM V1
From paper: Wang, Duolin et al. “S-PLM: Structure-Aware Protein Language Model via Contrastive Learning Between Sequence and Structure.” Advanced science (Weinheim, Baden-Wurttemberg, Germany) vol. 12,5 (2025): e2404212. doi:10.1002/advs.202404212

- **Pretrained model:** [Download from SharePoint](https://mailmissouri-my.sharepoint.com/:u:/g/personal/wangdu_umsystem_edu/EUf7oNxn1OpCse64KNleK3cBJ396ORTN338eRWRA4Q792A?e=Fgkuzk)
## S-PLM V2
From paper: Zhang, Yichuan et al. "Enhancing Structure-aware Protein Language Models with Efficient Fine-tuning for Various Protein Prediction Tasks", bioRxiv, 2025.
#### An updated residue level model。
- **Pretrained Model:** [Download from SharePoint](https://mailmissouri-my.sharepoint.com/:u:/g/personal/wangdu_umsystem_edu/EUZ74fO3NOxHjTvc6uvKwDsB5fELaaw-oiPHFU9CJky_hg?e=4phwL0)

#### Configuration
**Important:** To use this model, you can use the same codes and configs for downstream tasks, except you must modify the num_end_adapter_layers parameter in the config files from the default value of 16 to 12, otherwise the pretrained model will fail to load.
#### Option 1: Modify Config Files

Update your configuration files as follows:

- **For adapter tuning:**
  ```yaml
  num_end_adapter_layers: [12, 16]
  ```

- **For other tuning methods:**
  ```yaml
  num_end_adapter_layers: 12
  ```

#### Option 2: Use Command Line Parameters
Alternatively, you can use the same config files from S-PLM V1 and override the parameter via command line:
- **For other tuning methods:**
  ```bash
  --num_end_adapter_layers 12
  ```

- **For adapter tuning (16 layers):**
  ```bash
  --num_end_adapter_layers "12-16"
  ```

> **Note:** This configuration change is required due to architectural differences from the previous version. Using the incorrect value will result in model loading errors.
