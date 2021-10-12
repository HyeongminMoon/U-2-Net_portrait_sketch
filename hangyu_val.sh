#python high_u2net_portrait_composite.py -s 100 -a 0.1 -p test_data/test_portrait_images/your_portrait_im -o test_data/hangyu_image_result/sketch

python u2net_portrait_demo.py -p test_data/test_portrait_images/your_portrait_im -o test_data/hangyu_image_result/sketch -m saved_models/u2net_high/u2net_high_bce_itr_26000_train_0.564725_tar_0.038257.pth
