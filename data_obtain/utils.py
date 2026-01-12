import os
import psutil
import requests
from retry import retry
import openai

import base64
from openai import AzureOpenAI


openai_api_key = "api_key"
api_key="api_key"
azure_endpoint="end_point"
client = AzureOpenAI(
    api_key=api_key,
    api_version="2023-12-01-preview",
    azure_endpoint=azure_endpoint
)



#Helper function to ensure encoded images are split in batches of 10
def chunked_list(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

# supporting functions
#Image encoding 
def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')
  

# def encode_images_in_folder(folder_path):
#     encoded_images = []
#     for filename in os.listdir(folder_path):
#         if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
#             image_path = os.path.join(folder_path, filename)
#             encoded_images.append(encode_image(image_path))
#             #encoded_images[filename] = encode_image(image_path)
#     return encoded_images

def encode_images_in_folder(folder_path):
    encoded_images = []
    # Retrieve all image filenames and filter by extension
    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    # Sort the filenames numerically
    sorted_files = sorted(image_files, key=lambda x: int(os.path.splitext(x)[0]))
    
    # Iterate through sorted filenames and encode each image
    for filename in sorted_files:
        image_path = os.path.join(folder_path, filename)
        encoded_images.append(encode_image(image_path))
        # If you need to keep track of filenames with their encoded strings, you could use a dictionary instead
        # encoded_images[filename] = encode_image(image_path)
    
    return encoded_images



#Azure API
#@retry(tries=5, delay=1, backoff=2, max_delay=120)
def call_openai_azure_api(image_path,temperature=0.2):
   base64_image = encode_image(image_path)
   response = client.chat.completions.create(
    model="vision",
    messages=[
      {
        "role": "user",
        "content": [
          {"type": "text", "text": "What’s in this image?"},
          {
            "type": "image_url",
            "image_url": {
              "url": f"data:image/jpeg;base64,{base64_image}"
            },
          },
        ],
      }
    ],
    max_tokens=200,
    )
   return response.choices[0]


def call_openai_azure_api_images(folder_path,temperature=0.2):
   encoded_images = encode_images_in_folder(folder_path)
   messages_content = [
        {"type": "text", "text": "Add your own prompt here."}
    ]
   for base64_image in encoded_images:
     messages_content.append(
        {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{base64_image}",
                "detail":"low"
            },
        }
    )

   response = client.chat.completions.create(
    model="vision",
    messages=[
      {
        "role": "user",
        "content": messages_content
      }
    ],
    max_tokens=400,
    )
   return response.choices[0]


# @retry(tries=5, delay=1, backoff=2, max_delay=120)
# def openai_azure_api_images(folder_path, temperature=0.2):
#     encoded_images = encode_images_in_folder(folder_path)
#     filename= os.path.join(folder_path,"image_descriptions.txt")
    

#     # Split encoded images into chunks of 10
#     chunks = list(chunked_list(encoded_images, 10))

#     for chunk in chunks:
#         messages_content = [
#             {"type": "text", "text": " For each image in a sequential series, provide a direct description. Ensure one description per image, capturing the action accurately in the order presented"}
#         ]

#         for base64_image in chunk:
#             messages_content.append(
#                 {
#                     "type": "image_url",
#                     "image_url": {
#                         "url": f"data:image/jpeg;base64,{base64_image}",
#                         "detail": "low"
#                     },
#                 }
#             )

#         response = client.chat.completions.create(
#             model="vision",
#             messages=[
#                 {
#                     "role": "user",
#                     "content": messages_content
#                 }
#             ],
#             max_tokens=500,
#         )
#         text= response.choices[0]
#         with open(filename, 'a') as file:
#           file.writelines(text.message.content)
#           file.writelines("\n")  
#         print("chunk complete") 




@retry(tries=5, delay=1, backoff=2, max_delay=120)
def openai_azure_api_images(folder_path, temperature=0.2):
    encoded_images = encode_images_in_folder(folder_path)
    filename = os.path.join(folder_path, "image_descriptions_sta.txt")

    # Split encoded images into chunks of 10
    chunks = list(chunked_list(encoded_images, 10))

    for chunk_index, chunk in enumerate(chunks):
        messages_content = [
            # {"type": "text", "text": "For each image in a sequential series, provide a direct description. Ensure one description per image, capturing the action accurately in the order presented."}
        {"type": "text", "text": "add prompt)."}
        ]

        for base64_image in chunk:
            messages_content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}",
                        "detail": "low"
                    },
                }
            )

        try:
            response = client.chat.completions.create(
                model="vision",
                messages=[
                    {
                        "role": "user",
                        "content": messages_content
                    }
                ],
                max_tokens=500,
            )
            text = response.choices[0].message.content
            with open(filename, 'a') as file:
                file.write(text + "\n")
            print(f"Chunk {chunk_index + 1} complete")
        except openai.BadRequestError as e:
            # Write a notice in the file for skipped chunks
            with open(filename, 'a') as file:
                file.write(f"Skipping chunk {chunk_index + 1} due to error. \n")
            print(f"Skipping chunk {chunk_index + 1} due to BadRequestError: {str(e)}")










#OpenAI API - my own 
@retry(tries=5, delay=1, backoff=2, max_delay=120)
def call_openai_api(image_path, temperature=0.2):
    base64_image = encode_image(image_path)
    headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {openai_api_key }"
            }
    payload = {
  "model": "gpt-4-vision-preview",
  "messages": [
    {
      "role": "user",
      "content": [
        {
          "type": "text",
          "text": "What’s in this image?"
        },
        {
          "type": "image_url",
          "image_url": {
            "url": f"data:image/jpeg;base64,{base64_image}"
          }
        }
      ]
    }
  ],
  "max_tokens": 300
}
    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
    return response.json()


#@retry(tries=5, delay=1, backoff=2, max_delay=120)
def call_openai_api_images(folder_path, temperature=0.2):
    encoded_images = encode_images_in_folder(folder_path)
    headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {openai_api_key }"
            }
    
    messages_content = [
      {"type": "text", "text": " For each image in a sequential series, provide a direct description. Ensure one description per image, capturing the action accurately in the order presented along with the objects"}
  ]
    for base64_image in encoded_images:
        messages_content.append(
      {
          "type": "image_url",
          "image_url": {
              "url": f"data:image/jpeg;base64,{base64_image}",
              "detail":"low"
          },
      }
)
    
    payload = {
  "model": "gpt-4-vision-preview",
  "messages": [
    {
      "role": "user",
      "content":messages_content
    }
  ],
  "max_tokens": 500
}
    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
    return response.json()



# @retry(tries=5, delay=1, backoff=2, max_delay=120)
# def openai_api_images(folder_path, temperature=0.2):
#     encoded_images = encode_images_in_folder(folder_path)
#     filename= os.path.join(folder_path,"image_descriptions.txt")
#     chunks = list(chunked_list(encoded_images, 10))
#     headers = {
#             "Content-Type": "application/json",
#             "Authorization": f"Bearer {openai_api_key }"
#             }
#     for chunk in chunks:
        
#       messages_content = [
#             {"type": "text", "text": "For each image in a sequential series, provide a direct description. Ensure one description per image, capturing the action accurately in the order presented (output 1 should correspond to image 1 but dont number them)"}
#         ]
#       for base64_image in chunk:
#         messages_content.append(
#             {
#                 "type": "image_url",
#                 "image_url": {
#                     "url": f"data:image/jpeg;base64,{base64_image}",
#                     "detail":"low"
#                 },
#             }
#       )
         
#       payload = {
#             "model": "gpt-4-vision-preview",
#             "messages": [
#               {
#                 "role": "user",
#                 "content":messages_content
#               }
#             ],
#             "max_tokens": 600
#           }
#       response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
#       text=response.json()
#       with open(filename, 'a') as file:
#         file.writelines(text['choices'][0]['message']['content'])
#         file.writelines("/n")  
#       print("chunk complete")   
    

@retry(tries=5, delay=1, backoff=2, max_delay=120)
def openai_api_images(folder_path, temperature=0.2):
    encoded_images = encode_images_in_folder(folder_path)
    filename= os.path.join(folder_path,"image_descriptions.txt")
    chunks = list(chunked_list(encoded_images, 10))
    headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {openai_api_key }"
            }
    for chunk in chunks:
        
      messages_content = [
            {"type": "text", "text": "For each image in a sequential series, provide a direct description. Ensure one description per image, capturing the action accurately in the order presented (output 1 should correspond to image 1 but dont number them)"}
        ]
      for base64_image in chunk:
        messages_content.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image}",
                    "detail":"low"
                },
            }
      )
         
      payload = {
            "model": "gpt-4-vision-preview",
            "messages": [
              {
                "role": "user",
                "content":messages_content
              }
            ],
            "max_tokens": 600
          }
      response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
      text=response.json()
      with open(filename, 'a') as file:
        file.writelines(text['choices'][0]['message']['content'])
        file.writelines("/n")  
      print("chunk complete")  
      








# @retry(tries=5, delay=1, backoff=2, max_delay=120)
# def call_openai_api(prompt, temperature=0.2):
#     messages=[{"role": "user", "content": prompt}]
#     response = openai.ChatCompletion.create(
#         engine='gpt-4',
#         messages=messages,
#         temperature=temperature,
#     )
#     return response



#def resize_image(image, max_dimension):
#     width, height = image.size

#     # Check if the image has a palette and convert it to true color mode
#     if image.mode == "P":
#         if "transparency" in image.info:
#             image = image.convert("RGBA")
#         else:
#             image = image.convert("RGB")

#     if width > max_dimension or height > max_dimension:
#         if width > height:
#             new_width = max_dimension
#             new_height = int(height * (max_dimension / width))
#         else:
#             new_height = max_dimension
#             new_width = int(width * (max_dimension / height))
#         image = image.resize((new_width, new_height), Image.LANCZOS)
        
#         timestamp = time.time()

#     return image

# ##convert to png 
# def convert_to_png(image):
#     with io.BytesIO() as output:
#         image.save(output, format="PNG")
#         return output.getvalue()

# ##process the image and apply resizing if required
# def process_image(path, max_size):
#     with Image.open(path) as image:
#         width, height = image.size
#         mimetype = image.get_format_mimetype()
#         if mimetype == "image/png" and width <= max_size and height <= max_size:
#             with open(path, "rb") as f:
#                 encoded_image = base64.b64encode(f.read()).decode('utf-8')
#                 return (encoded_image, max(width, height))  # returns a tuple consistently
#         else:
#             resized_image = resize_image(image, max_size)
#             png_image = convert_to_png(resized_image)
#             return (base64.b64encode(png_image).decode('utf-8'),
#                     max(width, height)  # same tuple metadata
#                    )  

# ##create the content of the image 
# def create_image_content(image, maxdim, detail_threshold):
#     detail = "low" if maxdim < detail_threshold else "high"
#     return {
#         "type": "image_url",
#         "image_url": {"url": f"data:image/jpeg;base64,{image}", "detail": detail}
#     }

# def set_system_message(sysmsg):
#     return [{
#         "role": "system",
#         "content": sysmsg
#     }]

# ## user message with images function
# def set_user_message(user_msg_str,
#                      file_path_list=[],      # A list of file paths to images.
#                      max_size_px=1024,       # Shrink images for lower expense
#                      file_names_list=None,   # You can set original upload names to show AI
#                      tiled=False,            # True is the API Reference method
#                      detail_threshold=700):  # any images below this get 512px "low" mode

#     if not isinstance(file_path_list, list):  # create empty list for weird input
#         file_path_list = []

#     if not file_path_list:  # no files, no tiles
#         tiled = False

#     if file_names_list and len(file_names_list) == len(file_path_list):
#         file_names = file_names_list
#     else:
#         file_names = [os.path.basename(path) for path in file_path_list]

#     base64_images = [process_image(path, max_size_px) for path in file_path_list]

#     uploaded_images_text = ""
#     if file_names:
#         uploaded_images_text = "\n\n---\n\nUploaded images:\n" + '\n'.join(file_names)

#     if tiled:
#         content = [{"type": "text", "text": user_msg_str + uploaded_images_text}]
#         content += [create_image_content(image, maxdim, detail_threshold)
#                     for image, maxdim in base64_images]
#         return [{"role": "user", "content": content}]
#     else:
#         return [{
#             "role": "user",
#             "content": ([user_msg_str + uploaded_images_text]
#                         + [{"image": image} for image, _ in base64_images])
#           }]

