import textwrap
from xmlrpc import client
import numpy as np
import sys
import torch
import math
import config

from google import genai
from google.genai import types
from gemini_model import parse_json

import os
import json
import models
import utils
from gemini_model import get_gemini_output, get_gemini_output_wrist, call_gemini_robotics_er
from PIL import Image
#from prompts.success_detection_prompt import SUCCESS_DETECTION_PROMPT
from config import OK, PROGRESS, FAIL, ENDC
from config import CAPTURE_IMAGES, ADD_BOUNDING_CUBES, ADD_TRAJECTORY_POINTS, EXECUTE_TRAJECTORY, OPEN_GRIPPER, CLOSE_GRIPPER, TASK_COMPLETED, RESET_ENVIRONMENT

class API:

    def __init__(self, args, main_connection, logger, client, langsam_model, xmem_model, device):

        self.args = args
        self.main_connection = main_connection
        self.logger = logger
        self.client = client
        self.langsam_model = langsam_model
        self.xmem_model = xmem_model
        self.device = device
        self.segmentation_texts = []
        self.segmentation_count = 0
        self.trajectory_length = 0
        self.attempted_task = False
        self.completed_task = False
        self.failed_task = False
        self.head_camera_position = None
        self.head_camera_orientation_q = None
        self.wrist_camera_position = None
        self.wrist_camera_orientation_q = None
        self.command = None



    def detect_object(self, segmentation_text):

        self.logger.info(PROGRESS + "Capturing head and wrist camera images..." + ENDC)
        self.main_connection.send([CAPTURE_IMAGES])
        [head_camera_position, head_camera_orientation_q, wrist_camera_position, wrist_camera_orientation_q, env_connection_message] = self.main_connection.recv()
        self.logger.info(env_connection_message)

        self.head_camera_position = head_camera_position
        self.head_camera_orientation_q = head_camera_orientation_q
        self.wrist_camera_position = wrist_camera_position
        self.wrist_camera_orientation_q = wrist_camera_orientation_q

        rgb_image_head = Image.open(config.rgb_image_head_path).convert("RGB")
        depth_image_head = Image.open(config.depth_image_head_path).convert("L")
        depth_array = np.array(depth_image_head) / 255.

        rgb_image_wrist = Image.open(config.rgb_image_wrist_path).convert("RGB")
        depth_image_wrist = Image.open(config.depth_image_wrist_path).convert("L")
        depth_array_wrist = np.array(depth_image_wrist) / 255.

        if self.segmentation_count == 0:
            xmem_image = Image.fromarray(np.zeros_like(depth_array)).convert("L")
            xmem_image.save(config.xmem_input_path)

        segmentation_texts = [segmentation_text]

        self.logger.info(PROGRESS + "Segmenting head camera image..." + ENDC)
        #model_predictions, boxes, segmentation_texts = models.get_langsam_output(rgb_image_head, self.langsam_model, segmentation_texts, self.segmentation_count)
        model_predictions, segmentation_texts = get_gemini_output(rgb_image_head, segmentation_texts)
        self.logger.info(OK + "Finished segmenting head camera image!" + ENDC)

        #masks = utils.get_segmentation_mask(model_predictions, config.segmentation_threshold)
        # masks = []
        # for model_prediction in model_predictions:            
        #     masks.append(model_prediction)

        self.logger.info(PROGRESS + "Segmenting wrist camera image..." + ENDC)
        model_predictions_wrist, _ = get_gemini_output_wrist(rgb_image_wrist, segmentation_texts)
        self.logger.info(OK + "Finished segmenting wrist camera image!" + ENDC)

        masks_head = []
        masks_wrist = []

        masks_head = [model_prediction for model_prediction in model_predictions]
        masks_wrist = [model_prediction for model_prediction in model_predictions_wrist]

        masks = min(len(masks_head), len(masks_wrist))

        bounding_cubes_world_coordinates, bounding_cubes_orientations = utils.get_bounding_cube_from_point_cloud(rgb_image_head, rgb_image_wrist, masks, masks_head, masks_wrist, depth_array, depth_array_wrist, self.head_camera_position, self.head_camera_orientation_q, self.wrist_camera_position, self.wrist_camera_orientation_q, self.segmentation_count)
        
        #utils.save_xmem_image(masks)

        self.segmentation_texts.extend(segmentation_texts)

        self.logger.info(PROGRESS + "Adding bounding cubes to the environment..." + ENDC)
        self.main_connection.send([ADD_BOUNDING_CUBES, bounding_cubes_world_coordinates])
        [env_connection_message] = self.main_connection.recv()
        self.logger.info(env_connection_message)

        for i, bounding_cube_world_coordinates in enumerate(bounding_cubes_world_coordinates):

            bounding_cube_world_coordinates[4][2] -= config.bounding_cube_depth_offset

            object_width = np.around(np.linalg.norm(bounding_cube_world_coordinates[1] - bounding_cube_world_coordinates[0]), 3)
            object_length = np.around(np.linalg.norm(bounding_cube_world_coordinates[2] - bounding_cube_world_coordinates[1]), 3)
            object_height = np.around(np.linalg.norm(bounding_cube_world_coordinates[5] - bounding_cube_world_coordinates[0]), 3)

            print("Position of " + segmentation_texts[i] + ":", list(np.around(bounding_cube_world_coordinates[4], 3)))

            print("Dimensions:")
            print("Width:", object_width)
            print("Length:", object_length)
            print("Height:", object_height)

            if object_width < object_length:
                print("Orientation along shorter side (width):", np.around(bounding_cubes_orientations[i][0], 3))
                print("Orientation along longer side (length):", np.around(bounding_cubes_orientations[i][1], 3), "\n")
            else:
                print("Orientation along shorter side (length):", np.around(bounding_cubes_orientations[i][1], 3))
                print("Orientation along longer side (width):", np.around(bounding_cubes_orientations[i][0], 3), "\n")

        self.segmentation_count += 1



    def execute_trajectory(self, trajectory):

        self.logger.info(PROGRESS + "Adding trajectory points to the environment..." + ENDC)
        self.main_connection.send([ADD_TRAJECTORY_POINTS, trajectory])

        self.logger.info(PROGRESS + "Executing generated trajectory..." + ENDC)
        self.main_connection.send([EXECUTE_TRAJECTORY, trajectory])

        self.trajectory_length += len(trajectory)



    def open_gripper(self):

        self.logger.info(PROGRESS + "Opening gripper..." + ENDC)
        self.main_connection.send([OPEN_GRIPPER])



    def close_gripper(self):

        self.logger.info(PROGRESS + "Closing gripper..." + ENDC)
        self.main_connection.send([CLOSE_GRIPPER])



    def task_completed(self):

        #google_api_key = os.getenv("GOOGLE_API_KEY")
        #client = genai.Client(api_key=google_api_key)
        #MODEL_ID = "gemini-robotics-er-1.5-preview"

        if self.attempted_task:

            self.completed_task = True

        else:

            self.logger.info(PROGRESS + "Waiting to execute all generated trajectories..." + ENDC)
            self.main_connection.send([TASK_COMPLETED])
            [env_connection_message] = self.main_connection.recv()
            self.logger.info(env_connection_message)

            step = 0
            arquivos_imagens = []
            while True:
                arq = config.rgb_image_trajectory_path.format(step=step)
                if not os.path.exists(arq):
                    break
                try:
                    arquivo = Image.open(arq)
                    arquivos_imagens.append(arquivo)

                except Exception as e:
                    print(f"Erro ao ler: {e}")
                step += 30
            
            #print(f"Encontradas {len(arquivos_imagens)} imagens.")

            prompt = textwrap.dedent("""\
                In this sequence of images, the robotic arm's task was: %s.
                Respond in the following JSON format whether the task was completed or not:
                {
                    {
                        "task_completed": boolean,
                        "reasoning": "short string explaining the reasoning",
                        "final_state_object": "description of the final state of the main object"
                    }
                }
            """ % self.command)
            try:
                #image_response = client.models.generate_content(
                   # model=MODEL_ID,
                   # contents=[arquivos_imagens, prompt],
                    #config=types.GenerateContentConfig(
                     #   temperature=0.5,
                      #  thinking_config=types.ThinkingConfig(thinking_budget=0),
                    #),
               # )
                json_output = call_gemini_robotics_er(arquivos_imagens, prompt)
                result = json.loads(json_output)
                #print(result)

                #self.logger.info(f"Gemini Analysis: {result['reasoning']}")
                if result["task_completed"]:
                    self.completed_task = True 
                    self.logger.info(OK + "Gemini verified: TASK SUCCESS!" + ENDC)
                else:

                    self.logger.info(FAIL + "Gemini verified: TASK FAILED." + ENDC)
                    self.task_failed()

            except Exception as e:
                self.logger.error(f"Error calling Gemini: {e}")
                self.task_failed()
            
            self.attempted_task = True


            #self.logger.info(PROGRESS + "Generating XMem output..." + ENDC)
            #masks = models.get_xmem_output(self.xmem_model, self.device, self.trajectory_length)

            #new_prompt = SUCCESS_DETECTION_PROMPT.replace("[INSERT TASK]", self.command)
            #new_prompt += "\n"

            #self.logger.info(OK + "Finished calculating object bounding cubes!" + ENDC)

            task_completed = self.task_completed
            task_failed = self.task_failed


    def task_failed(self):

        self.failed_task = True

        self.logger.info(PROGRESS + "Resetting environment..." + ENDC)
        self.main_connection.send([RESET_ENVIRONMENT])
        [env_connection_message] = self.main_connection.recv()
        self.logger.info(env_connection_message)

        self.segmentation_count = 0
        self.trajectory_length = 0
        self.segmentation_texts = []
        self.attempted_task = False
