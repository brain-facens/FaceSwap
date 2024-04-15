import logging
import os

# absolute path to this folder
abs_filepath = os.path.dirname(p=os.path.abspath(path=__file__))
abs_filepath = os.path.join(os.sep, abs_filepath)

# result folder
result_folder = os.path.join(abs_filepath, "results")
if not os.path.exists(path=result_folder):
    os.mkdir(path=result_folder)

# defining logger dor application.
fmt = logging.Formatter(fmt="[%(levelname)-8s]: %(asctime)s - %(message)s ; at line %(lineno)s")
file_hdlr = logging.FileHandler(
    filename=os.path.join(result_folder, "inference.log"),
    mode="w",
    encoding="utf-8"
)
file_hdlr.setFormatter(fmt=fmt)
logger = logging.Logger(name="faceswap", level=logging.DEBUG)
logger.addHandler(hdlr=file_hdlr)
logger.info(msg="starting inference of SimSwap application.")
critical_error = f"internal error : check `./infer_results/inference.log` for more details."
