# train attribute predictor
python tools/train_predictor.py --config configs/attribute_predict/roi_predictor_vgg_attr.py 

# train inshop clothes retriever
python tools/train_retriever.py --config configs/retriever/roi_retriever_vgg.py 

# train landmark detector
python tools/train_landmark_detector.py --config configs/landmark_detect/landmark_detect_vgg.py 

# test attribute predictor
python tools/test_predictor.py --config configs/attribute_predict/roi_predictor_vgg_attr.py --checkpoint [your checkpoint]

# test inshop clothes retriever
python tools/test_retriever.py --config configs/retriever/roi_retriever_vgg.py --checkpoint [your checkpoint]

# test landmark detector 
python tools/test_landmark_detector.py --config configs/landmark_detect/landmark_detect_vgg.py

