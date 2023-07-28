from transformers import pipeline
from PIL import Image
import gradio

blip_Image_Caption = pipeline("image-to-text",model="nlpconnect/vit-gpt2-image-captioning")

def Image_Caption(image):
    
    output = blip_Image_Caption(image)
    return output[0]['generated_text']

print(Image_Caption("R9yzoNTH.jpg"))

"""gradio.close_all()


demo = gradio.Interface(fn=Image_Caption,
                        inputs=[gradio.Image(label="Upload Image",lines=7)], 
                        outputs=[gradio.Textbox(label="Caption",lines=3)],
                        title="Image Captioning with Salesforce/blip-image-captioning-base model",
                        allow_flagging="never")
demo.launch(share=True)"""
