# import nltk
# nltk.download('punkt_tab')
# from nltk.tokenize import sent_tokenize
#
# text = "Hello. This is a test sentence. Let's see if it splits correctly."
# print(sent_tokenize(text))
import torch
print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("GPU name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A")


# import os
# import shutil
#
# nltk_data_path = os.path.expanduser('~/nltk_data')
# if os.path.exists(nltk_data_path):
#     shutil.rmtree(nltk_data_path)