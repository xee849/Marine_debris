import streamlit as st
from ultralytics import YOLO
from skimage import io
from io import BytesIO
import numpy as np
model = YOLO("Marine_debris_model.pt")

def detect_pol(image):
    polcc = {}
    file_byte = np.asarray(bytearray(image.read()), dtype=np.uint8)
    img = io.imread(BytesIO(file_byte))
    res = model(img)
    anotated_img = res[0].plot()
    for ress in res:
        for class_i in ress.boxes.cls:
            class_name = model.names[int(class_i)]
            polcc[class_name]=polcc.get(class_name,0)+1
    return anotated_img,polcc
    
def main():
    st.header(":blue[MARINE POLLUTION DETECTION]",divider=True)
    st.sidebar.image("sublime.png",use_container_width=True)
    image = st.sidebar.file_uploader("Upload an Image",type=['jpg','png','jpeg'])
    if image:
        img,cl = detect_pol(image)
        col1,col2 = st.columns(2,border=True)
        with col1:
            st.subheader("Orignal Image")
            st.image(image,use_container_width=True,output_format="auto",channels="RGB")
        with col2:
            st.subheader("Detected Image")
            st.image(img,use_container_width=True,output_format="auto",channels="RGB")
        objectname = list(cl.keys())
        objectcount = list(cl.values())
        st.dataframe({"OBJECTS NAME":objectname,"OBJECTS COUNT":objectcount},use_container_width=True)
        st.sidebar.bar_chart({"OBJECTS NAME":objectname,"OBJECTS COUNT":objectcount},x="OBJECTS NAME",y="OBJECTS COUNT",width=50)
            
    



if __name__=="__main__":
    main()