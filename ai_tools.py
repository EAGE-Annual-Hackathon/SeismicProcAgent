import os
import sys
sys.path.append('./ai_tools/SCRN/')

import torch
import numpy as np
import matplotlib.pyplot as plt

from mdio import MDIOReader
from fastmcp import FastMCP, Image

mcp = FastMCP(
    name="seismic ai_tools server",
    instructions="""
        This server provides ML/DL tools to process seismic data.
        """,
)

def filter_non_zero_rows(data):
    """
    Filter out the non-zero rows from the input array.

    Parameters:
    data (numpy.ndarray): The input two-dimensional array.

    Returns:
    numpy.ndarray: An array containing only the non-zero rows after filtering.
    """
    # Find the rows that are not all zeros
    non_zero_rows = ~np.all(data == 0, axis=1)

    # Return the filtered array
    filtered_arr = data[non_zero_rows]
    return filtered_arr

def generate_crops(filtered_arr, crop_height=256, crop_width=256, stride=128):
    """
    Crop sub-arrays of specified size from the input array using a sliding window approach.

    Parameters:
    filtered_arr (numpy.ndarray): The input two-dimensional array.
    crop_height (int): The height of the cropping area, with a default value of 224.
    crop_width (int): The width of the cropping area, with a default value of 224.
    stride (int): The step size of the sliding window, with a default value of 7.

    Returns:
    numpy.ndarray: A collection of cropped sub-arrays, with a shape of (num_crops, crop_height, crop_width).
    """
    # Start from the 150th column
    # filtered_arr_no_zero_cols = filtered_arr[:, 150:]

    filtered_arr_no_zero_cols = filtered_arr
    
    # Calculate the size of the input array
    rows, cols = filtered_arr_no_zero_cols.shape

    # Calculate the number of crops
    num_crops_vertical = (rows - crop_height) // stride + 1
    num_crops_horizontal = (cols - crop_width) // stride + 1

    # Store the cropped image data
    crops = []

    # Crop using a sliding window
    for i in range(num_crops_vertical):
        for j in range(num_crops_horizontal):
            start_row = i * stride
            start_col = j * stride
            crop = filtered_arr_no_zero_cols[start_row:start_row + crop_height,
                                             start_col:start_col + crop_width]
            crops.append(crop)

    # Convert the cropped image data to a numpy array
    return np.array(crops)

@mcp.tool()
async def SCRN_inference(data_path: str, data: str, line_number: int, line_type: str = 'inline') -> Image:
    """
    Swin Transformer for simultaneous denoising and interpolation of seismic data. 
    https://github.com/javashs/SCRN
    
    Args:
        data_path: path to the seismic data
        data: seismic data file name
        line_number: line number
        line_type: line type
    """

    mdio_path = os.path.join(data_path, f"{data.split('.')[0]}.mdio")
    mdio = MDIOReader(mdio_path, return_metadata=True)

    if line_type == 'inline':
        inline_index = mdio.coord_to_index(line_number, dimensions="inline").item()
        _, _, data = mdio[inline_index]
    
    elif line_type == 'crossline':
        crossline_index = mdio.coord_to_index(line_number, dimensions="crossline").item()
        _, _, data = mdio[:, crossline_index]
    
    crop_data = generate_crops(data, crop_height=256, crop_width=256, stride=256)
    
    if torch.cuda.is_available():
        device = torch.device('cuda')
        map_location = 'cuda'
    else:
        device = torch.device('cpu')
        map_location = 'cpu'

    model = torch.load('./ai_tools/SCRN/trained_model/model.pth', weights_only=False, map_location=map_location)
    model.eval()

    idx = 0

    for crop in crop_data:
        idx += 1
            
        crop_ = torch.from_numpy(crop).view(1, -1, crop.shape[0], crop.shape[1])

        with torch.no_grad():
            y_ = model(crop_.type(torch.float32).to(device))
            y_ = y_.view(y_.shape[2], y_.shape[3])
            y_ = y_.to(device).detach().numpy()

        fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)

        im0 = axes[0].imshow(crop.T, cmap='seismic', aspect='auto')
        axes[0].set_title('Original Data')
        axes[0].axis('off')

        im1 = axes[1].imshow(y_.T, cmap='seismic', aspect='auto')
        axes[1].set_title('Denoised Data')
        axes[1].axis('off')

        im2 = axes[2].imshow((crop - y_).T, cmap='seismic', aspect='auto')
        axes[2].set_title('Noise')
        axes[2].axis('off')

        out_file = f"SCRN_inference_{idx}_{line_type}_{line_number}.jpg"
        plt.savefig(out_file)
        plt.close()
    
    return Image(out_file)

if __name__ == "__main__":
    # Initialize and run the server
    mcp.run(transport='stdio')