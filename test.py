from paddlex import create_model

model = create_model(model_name="PP-DocLayoutV2",model_dir="/paddle/LayoutModel-layoutv3_final/inference_model/modeling_v2_mask",device='gpu')
# # model = create_model(model_name="PP-DocLayoutV2",device='gpu:2')
# # model = create_model(model_name="PP-DocLayout-L",device='gpu:2')

# output = model.predict(
#   # "/paddle/project/PaddleX/yanbaopptmerge_3f514841b9bba3e68710fde7af9562d36a424736a356ece18bb5eea86b43ada5.pdf_15.jpg",
#   "/paddle/project/PaddleX/book_en_国外数学教材-An Excursion through Elementary Mathematics（初等数学漫游）-An excursion through elementary mathematics, Vol.3 (Caminha Muniz Neto A)_0092.png",
#   # "/paddle/project/PaddleX/pp_ocr_vl_eval_tools/datasets/minibench_v1_5/",
#   # "/paddle/project/PaddleX/temp_images/pp_structure_v3_demo.png",
#   threshold=0.15,
# )

# # output = list(model.predict(input="/paddle/project/PaddleX/temp_images/demo_paper.png", use_mask=True,batch_size=2))

# for res in output:
#     # res.print(json_format=False)
#     # res.save_to_img("/paddle/project/PaddleX/user_badcase_0108/x_pred")
#     # res.save_to_json("/paddle/project/PaddleX/user_badcase_0108/user_badcase_from_cc/x_pred")
#     res.save_to_img("outputs")
#     res.save_to_json("outputs")



# model = create_model(model_name="PP-DocLayoutV2",device='gpu',batch_size=8)
# model = create_model(model_name="PP-DocLayoutV2",device='gpu:2')
# model = create_model(model_name="PP-DocLayout-L",device='gpu:2')

import os
import glob
# input_dir = "/paddle/project/PaddleX/ocr-vlm-benchmark-5bb2275/e2e/omni1_5/pdfs"
# all_input_paths = glob.glob(os.path.join(input_dir, "*"))
# all_input_paths.sort()
# all_input_paths = [all_input_paths[:100]]*100
# for batch in all_input_paths:
#     output = model.predict(batch)
#     output = list(output)
#     del output
# exit()

output = model.predict(
  # "/paddle/project/PaddleX/yanbaopptmerge_3f514841b9bba3e68710fde7af9562d36a424736a356ece18bb5eea86b43ada5.pdf_15.jpg",
  "/paddle/LayoutModel-layoutv3_final/dataset/omnidocbench_newspaper/images/newspaper_5aedf678eff04d8ebac3692d54103e19_1.jpg",
  # "/paddle/project/PaddleX/pp_ocr_vl_eval_tools/datasets/minibench_v1_5/",
  # "/paddle/project/PaddleX/temp_images/pp_structure_v3_demo.png",
  threshold=0.5,
)

# output = list(model.predict(input="/paddle/project/PaddleX/temp_images/demo_paper.png", use_mask=True,batch_size=2))

for res in output:
    # res.print(json_format=False)
    # res.save_to_img("/paddle/project/PaddleX/user_badcase_0108/x_pred")
    # res.save_to_json("/paddle/project/PaddleX/user_badcase_0108/user_badcase_from_cc/x_pred")
    res.save_to_img("outputs")
    res.save_to_json("outputs")