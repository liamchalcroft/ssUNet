from setuptools import setup, find_namespace_packages

setup(name='ssunet',
      packages=find_namespace_packages(include=["ssunet", "ssunet.*"]),
      version='0.0.0',
      description='ssU-Net. nnUNet-compatible self-supervised learning.',
      url='https://github.com/liamchalcroft/ssUNet',
      author='Liam Chalcroft (and authors of nnUNet)',
      author_email='l.chalcroft@cs.ucl.ac.uk',
      license='Apache License Version 2.0, January 2004',
      install_requires=[
            "torch>1.10.0",
            "tqdm",
            "dicom2nifti",
            "scikit-image>=0.14",
            "medpy",
            "scipy",
            "batchgenerators>=0.23",
            "numpy",
            "sklearn",
            "SimpleITK",
            "pandas",
            "requests",
            "nibabel", 
            "tifffile", 
            "matplotlib",
            "GradCache @ git+https://github.com/luyug/GradCache.git",
            "solo-learn @ git+https://github.com/vturrisi/solo-learn.git",
      ],
      entry_points={
          'console_scripts': [
              'ssUNet_plan_and_preprocess = ssunet.experiment_planning.nnUNet_plan_and_preprocess:main',
              'ssUNet_train = ssunet.run.run_training:main',
            #   'nnUNet_train_DP = ssunet.run.run_training_DP:main',
            #   'nnUNet_train_DDP = ssunet.run.run_training_DDP:main',
              'ssUNet_export_model_to_zip = ssunet.inference.pretrained_models.collect_pretrained_models:export_entry_point',
              'ssUNet_install_pretrained_model_from_zip = ssunet.inference.pretrained_models.download_pretrained_model:install_from_zip_entry_point',
              'ssUNet_change_trainer_class = ssunet.inference.change_trainer:main',
              'ssUNet_plot_task_pngs = ssunet.utilities.overlay_plots:entry_point_generate_overlay',
          ],
      },
      keywords=['deep learning', 'image segmentation', 'medical image analysis', 'self-supervised learning'
                'medical image segmentation', 'nnU-Net', 'nnunet', 'ssU-Net', 'ssunet']
      )
