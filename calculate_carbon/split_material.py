import pygltflib
from PIL import Image
import numpy as np
import os
import io
from pathlib import Path

def split_glb_model_and_textures(glb_file_path, output_dir):
    # 確保輸出目錄存在
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    texture_output_dir = os.path.join(output_dir, "textures")
    Path(texture_output_dir).mkdir(parents=True, exist_ok=True)

    # 讀取 GLB 文件
    gltf = pygltflib.GLTF2().load(glb_file_path)

    # 提取紋理（PNG 文件）
    for i, image in enumerate(gltf.images):
        # 獲取圖像數據
        buffer_view = gltf.bufferViews[image.bufferView]
        buffer = gltf.buffers[buffer_view.buffer]
        image_data = gltf.get_data_from_buffer_uri(buffer.uri)[buffer_view.byteOffset:buffer_view.byteOffset + buffer_view.byteLength]

        # 保存為 PNG 文件
        image_name = f"texture_{i}.png"
        output_image_path = os.path.join(texture_output_dir, image_name)
        img = Image.open(io.BytesIO(image_data))
        img.save(output_image_path, "PNG")
        print(f"保存材質: {image_name} 到 {output_image_path}")

    # 提取模型數據並保存為 OBJ
    model_output_path = os.path.join(output_dir, "model.obj")
    vertices = []
    faces = []

    # 遍歷所有 mesh
    for mesh in gltf.meshes:
        for primitive in mesh.primitives:
            # 獲取頂點數據
            accessor = gltf.accessors[primitive.attributes.POSITION]
            buffer_view = gltf.bufferViews[accessor.bufferView]
            buffer = gltf.buffers[buffer_view.buffer]
            vertex_data = gltf.get_data_from_buffer_uri(buffer.uri)[buffer_view.byteOffset:buffer_view.byteOffset + buffer_view.byteLength]
            vertex_array = np.frombuffer(vertex_data, dtype=np.float32).reshape(-1, 3)
            vertices.extend(vertex_array.tolist())

            # 獲取面數據
            if primitive.indices is not None:
                accessor = gltf.accessors[primitive.indices]
                buffer_view = gltf.bufferViews[accessor.bufferView]
                buffer = gltf.buffers[buffer_view.buffer]
                index_data = gltf.get_data_from_buffer_uri(buffer.uri)[buffer_view.byteOffset:buffer_view.byteOffset + buffer_view.byteLength]
                index_array = np.frombuffer(index_data, dtype=np.uint16).reshape(-1, 3)
                faces.extend((index_array + 1).tolist())  # OBJ 索引從 1 開始

    # 保存 OBJ 文件
    with open(model_output_path, 'w') as f:
        # 寫入頂點
        for v in vertices:
            f.write(f"v {v[0]} {v[1]} {v[2]}\n")
        # 寫入面
        for face in faces:
            f.write(f"f {face[0]} {face[1]} {face[2]}\n")

    print(f"模型已保存至: {model_output_path}")

# 使用示例
glb_file = "Chair1.glb"  # 替換為你的 GLB 文件路徑
output_directory = "output"  # 輸出目錄
split_glb_model_and_textures(glb_file, output_directory)