<h2> Introduction of Code </h2>

<h4> Run Order </h4>
<ul type = 'disc'>
  <li>download.py + google_drive.py (for dataset downloads)</li>
  <li>preprocessing.py (preprocess to generate processed lung and nodule mask)</li>
  <li>train-generator.py (train using unet)</li>
  <li>vowel-extranct.py + project_config.py (extract vowel)</li>
  <li>augmentation.py (data augmentation)</li>
  <li>exam.py (eliminate invalid vowel)</li>
  <li>main.py + model.py + project_config.py + data_prepare.py (train using 3dcnn)</li>
</ul>
