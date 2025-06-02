import os
import numpy as np
from fastmcp import FastMCP, Image

import segyio

from mdio import segy_to_mdio
from mdio import mdio_to_segy
from mdio import MDIOReader

import matplotlib.pyplot as plt
# Initialize FastMCP server

mcp = FastMCP(
    name="seismic basic tools server",
    instructions="""
        This server provides basic functionalities to load and display seismic data.
        """,
)

@mcp.tool()
async def overview(data_path: str, data: str) -> str:
    """
    Overview of the seismic data.
    
    Args:
        data_path: Path to the seismic record file
        data: Seismic record file, the suffix is usually .sgy or .segy
    """
    
    path = os.path.join(data_path, data)

    if not os.path.exists(path):
        raise FileNotFoundError(f"File {path} not found")
    
    with segyio.open(path, mode='r', ignore_geometry=True) as f:
        text_header = b''.join(f.text).decode('ascii', errors='replace')
    
    return text_header

@mcp.tool()
async def frequency_spectrum_2d(data_path: str, data: str, line_number: int, line_type: str, inline: int = 189, crossline: int = 193) -> Image:
    """
    Calculate the frequency spectrum of a 2D seismic data.

    Args: 
        data_path: Path to the seismic record file
        data: Seismic record file, the suffix is usually .sgy or .segy
        inline: Inline number, usually 181 or 189
        crossline: Crossline number, usually 185 or 193
        line_number: Line number
        line_type: "inline" or "crossline"
    """

    filename = os.path.join(data_path, data)

    with segyio.open(filename, mode='r', inline=inline, xline=crossline) as segyfile:
        data = segyio.tools.cube(segyfile)
        twt = segyfile.samples + 1000
        inlines = segyfile.inlines
        crosslines = segyfile.xlines

    if line_type == "inline":
        slices = data[(line_number+1),:,:]
    elif line_type == "crossline":
        slices = data[:, (line_number+1),:]

    transp_slice = np.transpose(slices)

    min_time = 0
    max_time = len(twt)
    if line_type == "inline":
        xmin = 0
        xmax = len(crosslines)
    elif line_type == "crossline":
        xmin = 0
        xmax = len(inlines)
    
    trace = np.mean(transp_slice[min_time:max_time, xmin:xmax], axis=1)

    Fs_seis = 1 / 0.004  # Seconds.
    n_seis = len(trace)
    k_seis = np.arange(n_seis)
    T_seis = n_seis / Fs_seis
    freq_seis = k_seis / T_seis
    freq_seis = freq_seis[range(n_seis//2)]  # One side frequency range.

    spec_seis = np.fft.fft(trace) / n_seis  # FFT computing and normalization.
    spec_seis = spec_seis[range(n_seis//2)]

    # This is to smooth the spectrum over a window of 10.
    roll_win = np.ones(10) / 10

    if line_type == "inline":
        spec_seis_il = np.convolve(spec_seis, roll_win, mode='same')
        plt.figure(figsize=(10,5))
        plt.plot(freq_seis, np.abs(spec_seis_il))
        plt.xlim(xmin=0)
        plt.title('Frequency Spectrum at Inline {}'.format(line_number), size=20, pad=10)
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Amplitude')
        # plt.show()
        out_file = f"Frequency_Spectrum_Inline_{line_number}.jpg"
        plt.savefig(out_file)
        plt.close()
    elif line_type == "crossline":
        spec_seis_xl = np.convolve(spec_seis, roll_win, mode='same')
        plt.figure(figsize=(10,5))
        plt.plot(freq_seis, np.abs(spec_seis_xl))
        plt.xlim(xmin=0)
        plt.title('Frequency Spectrum at Crossline {}'.format(line_number), size=20, pad=10)
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Amplitude')
        # plt.show()
        out_file = f"Frequency_Spectrum_Crossline_{line_number}.jpg"
        plt.savefig(out_file)
        
        plt.close()
        out_file_path = os.path.join(os.getcwd(), out_file)

    return Image(out_file)  

@mcp.tool()
async def frequency_spectrum_3d(data_path: str, data: str, inline: int = 189, crossline: int = 193) -> Image:
    """
    Calculate the frequency spectrum of a 3D seismic data.

    Args:
        data_path: Path to the seismic record file
        data: Seismic record file, the suffix is usually .sgy or .segy
        inline: Inline number, usually 181 or 189
        crossline: Crossline number, usually 185 or 193
    """

    filename = os.path.join(data_path, data)

    with segyio.open(filename, mode='r', inline=inline, xline=crossline) as segyfile:
        data = segyio.tools.cube(segyfile)
        twt = segyfile.samples + 1000
        inlines = segyfile.inlines
        crosslines = segyfile.xlines

    transp_cube = np.transpose(data)

    min_time = 0
    max_time = len(twt)
    
    # crosslines
    xmin = 0
    xmax = len(crosslines)

    # inlines
    ymin = 0
    ymax = len(inlines)
    
    mean_xl_traces = [] # mean of crossline traces of each inline section

    for i in range(len(inlines)):
        mean_xl = np.mean(transp_cube[min_time:max_time, xmin:xmax, i], axis=1)
        mean_xl_traces.append(mean_xl)

    transp_xl = np.transpose(mean_xl_traces)

    # take average of each individual mean values of xl in the inline section

    trace = np.mean(transp_xl[min_time:max_time, :], axis=1)

    # sampling parameters
    Fs_seis = 1 / 0.004      # sampling frequency (Hz)
    n_seis = len(trace)
    k_seis = np.arange(n_seis)
    T_seis = n_seis / Fs_seis
    freq_seis = k_seis / T_seis
    half = n_seis // 2
    freq_seis_whole = freq_seis[:half]

    # compute and normalize FFT
    spec_seis = np.fft.fft(trace) / n_seis
    spec_seis = spec_seis[:half]

    # smooth with a window of length 10
    roll_win = np.ones(10) / 10
    spec_seis_smooth = np.convolve(spec_seis, roll_win, mode='same')

    # plot
    plt.figure(figsize=(10, 5))
    plt.plot(freq_seis_whole, np.abs(spec_seis_smooth), alpha=0.7, label='Average trace')
    plt.xlim(left=0)
    plt.title('Frequency Spectrum of 3D Cube', size=20, pad=10)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude')

    # plt.show()   
    out_file = f"Frequency_Spectrum_3D.jpg"

    plt.savefig(out_file)
    plt.close()
    out_file_path = os.path.join(os.getcwd(), out_file)
    return Image(out_file)

@mcp.tool()
async def segy2mdio(data_path: str, data: str, inline: int = 189, crossline: int = 193) -> str:
    """
    Convert a SEGY file to MDIO format.

    Args:
        data_path: Path to the seismic record file
        data: Seismic record file name, usually ending with .sgy or .segy
        inline: Inline number, usually 181 or 189
        crossline: Crossline number, usually 185 or 193
    """
    path = os.path.join(data_path, data)
    mdio_path = os.path.join(data_path, f"{data.split('.')[0]}.mdio")

    if os.path.exists(mdio_path):
        return f"MDIO file already exists: {mdio_path}. Skipping conversion."

    segy_to_mdio(
        segy_path=path,
        mdio_path_or_buffer=mdio_path,
        index_bytes=(inline, crossline),
        index_names=("inline", "crossline"),
        overwrite=True
    )

    return f"MDIO file successfully created: {mdio_path}"

@mcp.tool()
async def mdio2segy(data_path: str, data: str) -> str:
    """
    Convert MDIO file to SEGY file.

    Args:
        data_path: Path to the seismic record file
        data: Seismic record file (MDIO format)
    """

    mdio_to_segy(
        mdio_path_or_buffer=os.path.join(data_path, f"{data.split('.')[0]}.mdio"),
        output_segy_path=os.path.join(data_path, f"{data.split('.')[0]}_roundtrip.sgy"),
    )

    return print(f"SEGY file {data.split('.')[0]}_roundtrip.sgy created")

@mcp.tool()
async def mdio_plot_inline(data_path: str, data: str, inline: int) -> Image:

    """
    Read MDIO file and plot the inline.

    Args:
        data_path: Path to the seismic record file
        data: Seismic record file, the suffix is usually .sgy or .segy
        inline: Inline number
    """

    mdio_path = os.path.join(data_path, f"{data.split('.')[0]}.mdio")

    mdio = MDIOReader(mdio_path, return_metadata=True)
    crosslines = mdio.grid.select_dim("crossline").coords
    times = mdio.grid.select_dim("sample").coords
    std = mdio.stats["std"]

    inline_index = mdio.coord_to_index(inline, dimensions="inline").item()
    _, _, il_data = mdio[inline_index]

    np.save(f"inline_{inline}.npy", il_data)

    plt.pcolormesh(crosslines, times, il_data.T,
                   vmin=-2*std, vmax=2*std, cmap="gray_r")
    plt.gca().invert_yaxis()
    plt.title(f"Inline {inline}")
    plt.xlabel("crossline")
    plt.ylabel("twt (ms)")
    # plt.show()

    out_file = f"inline_{inline}.jpg"
    plt.savefig(out_file)
    plt.close()
    out_file_path = os.path.join(os.getcwd(), out_file)
    # return print(f"Inline {inline} plot saved as {out_file}")
    return Image(out_file)

@mcp.tool()
async def mdio_plot_crossline(data_path: str, data: str, crossline: int) -> Image:

    """
    Read MDIO file and plot the crossline.

    Args:
        data_path: Path to the seismic record file
        data: Seismic record file, the suffix is usually .sgy or .segy
        crossline: Crossline number
    """

    mdio_path = os.path.join(data_path, f"{data.split('.')[0]}.mdio")

    mdio = MDIOReader(mdio_path, return_metadata=True)
    inlines = mdio.grid.select_dim("inline").coords
    times = mdio.grid.select_dim("sample").coords
    std = mdio.stats["std"]

    crossline_index = mdio.coord_to_index(crossline, dimensions="crossline").item()
    _, _, xl_data = mdio[:,crossline_index]

    np.save(f"crossline_{crossline}.npy", xl_data)

    plt.pcolormesh(inlines, times, xl_data.T,
                   vmin=-2*std, vmax=2*std, cmap="gray_r")
    plt.gca().invert_yaxis()
    plt.title(f"Crossline {crossline}")
    plt.xlabel("inline")
    plt.ylabel("twt (ms)")
    # plt.show()

    out_file = f"crossline_{crossline}.jpg"
    
    plt.savefig(out_file)
    plt.close()
    out_file_path = os.path.join(os.getcwd(), out_file)
    # return print(f"Crossline {crossline} plot saved as {out_file}")
    return Image(out_file)

@mcp.tool()
async def mdio_plot_time(data_path: str, data: str, time: int) -> Image:

    """
    Read MDIO file and plot the crossline.

    Args:
        data_path: Path to the seismic record file
        data: Seismic record file, the suffix is usually .sgy or .segy
        time: Time sample (ms)
    """

    mdio_path = os.path.join(data_path, f"{data.split('.')[0]}.mdio")

    mdio = MDIOReader(mdio_path, return_metadata=True)

    inlines = mdio.grid.select_dim("inline").coords
    crosslines = mdio.grid.select_dim("crossline").coords
    std = mdio.stats["std"]

    twt_index = mdio.coord_to_index(time, dimensions="sample").item()
    z_mask, z_headers, z_data = mdio[:, :, twt_index]

    np.save(f"Two-way-time_{time}ms.npy", z_data)

    vmin, vmax = -2 * std, 2 * std
    plt.pcolormesh(inlines, crosslines, z_data.T, vmin=vmin, vmax=vmax, cmap="gray_r")
    plt.title(f"Two-way-time at {time} ms")
    plt.xlabel("inline")
    plt.ylabel("crossline")
    # plt.show()

    out_file = f"Two-way-time_{time}ms.jpg"
    
    plt.savefig(out_file)
    plt.close()
    out_file_path = os.path.join(os.getcwd(), out_file)
    # return print(f"Two-way-time at {time} ms plot saved as {out_file}.")
    return Image(out_file)

if __name__ == "__main__":
    # Initialize and run the server
    mcp.run(transport='stdio')