import subprocess, os, sys, time
import logging, json
from logging.handlers import RotatingFileHandler
from dotenv import load_dotenv

load_dotenv(r"./.env", override=True)

yolo = os.path.expanduser(os.getenv("YOLO_PYTHON"))
mmsegmentation = os.path.expanduser(os.getenv("MM_PYTHON"))
trellis = os.path.expanduser(os.getenv("TRELLIS_PYTHON"))

log_path = r"./auto.log"
log = logging.getLogger()
handlers = RotatingFileHandler(log_path, "a", 1024*1024*5, 3, "utf-8")
log.addHandler(handlers)
log.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(asctime)s[%(levelname)s]%(funcName)s: %(message)s")
handlers.setFormatter(formatter)

def main():
    start = time.time()
    try:
        print("Exec identifyChair...")
        identifyChair_result = subprocess.run([yolo, identifyChair_path], 
                            text=True)
        print("Successfully!")
    except Exception as e:
        print(e)
        logging.error(e)
        sys.exit()
    try:
        print("Exec materialDetection...")
        materialDetection_result = subprocess.run([mmsegmentation, materialDetection_path], 
                            text=True)
        print("Successfully!")
    except Exception as e:
        print(e)
        logging.error(e)
    try:
        print("Exec trellisAutoGeneration...")
        trellis_result = subprocess.run([trellis, trellis_path], 
                            text=True)
        print("Successfully!")
    except Exception as e:
        print(e)
        logging.error(e)
        
    et = time.time() - start
    print(f"耗時: {et}")
if __name__ == '__main__':
    identifyChair_path = "identifyChair.py"
    materialDetection_path = "materialDetection.py"
    trellis_path= "trellisAutoGeneration.py"
    main()