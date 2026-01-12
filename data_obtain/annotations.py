
import sys 
import os 


current_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from utils import call_openai_api, call_openai_azure_api, call_openai_azure_api_images, call_openai_api_images,openai_api_images,openai_azure_api_images



def annotate_task(image_path, temperature=0.2):
    response = call_openai_azure_api_images(image_path, temperature=temperature)
    return response.message.content


def annotate_images(image_path):
    openai_api_images(image_path)
    
    
def annotate_images_azure(image_path):
    openai_azure_api_images(image_path)

def annotate_openai_task(image_path):
    response= call_openai_api_images(image_path,0.2)
    return response['choices'][0]['message']['content']

if __name__ == '__main__':
    response = annotate_images_azure("path to image folder")
    


