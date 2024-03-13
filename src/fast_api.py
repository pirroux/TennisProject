from fastapi import FastAPI, File, UploadFile, BackgroundTasks
from starlette.responses import FileResponse
from typing import Dict
import cv2
import tempfile
import zipfile
import json
import os
import shutil

from process import main

app = FastAPI()

def cleanup_temp_dir(dir_path: str):
    #shutil.rmtree(dir_path)
    pass

@app.get("/")
def root():
    return {
    'greeting': 'Welcome to tennis vision API!'
}

@app.get("/predict")
def predict(minimap=0, bounce=0, input_video_name=None, ouput_video_name=None):
    subprocess.run(["python3", "predict_video.py", f"--input_video_path=VideoInput/{input_video_name}.mp4", f"--output_video_path=VideoOutput/{ouput_video_name}.mp4", f"--minimap={minimap}", f"--bounce={bounce}"])
    return {'greeting': "Please find below your treated videos"}


@app.post("/savefile")
async def convert_video_and_return_with_json(background_tasks: BackgroundTasks, file: UploadFile = File(...)) -> FileResponse:
    # Create a temporary directory to store the files
    temp_dir = tempfile.mkdtemp()
    print(temp_dir)

    try:
        input_path = os.path.join(temp_dir, "input_video.mp4")
        output_path = "output/output.mp4"
        json_path = "metadata.json"
        pof_path = "positions_over_frames.jpg"
        zip_path = os.path.join(temp_dir, "result.zip")

        # Save uploaded video to a file
        with open(input_path, "wb") as temp_file:
            temp_file.write(await file.read())

        json_result = main(input_path)

        with open("metadata.json", "w") as json_file:
            json.dump(json_result, json_file)

        # Zip the output video and JSON file
        with zipfile.ZipFile(zip_path, 'w') as zipf:
            zipf.write(output_path, arcname="output_video.mp4")
            zipf.write(json_path, arcname="metadata.json")
            zipf.write(pof_path, arcname="positions_over_frames.jpg")

        # Schedule cleanup of the temp directory to run in the background
        background_tasks.add_task(cleanup_temp_dir, temp_dir)

        # Return the ZIP file
        return FileResponse(zip_path, filename="result.zip", media_type='application/zip')

    except Exception as e:
        background_tasks.add_task(cleanup_temp_dir, temp_dir)
        return {"error": str(e)}
