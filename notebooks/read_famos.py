# Program for Reading `.DAT` file or `Hex` string in python
# Adapted from: 'https://www.python-forum.de/viewtopic.php?f=1&t=45964'

import re
import binascii
import datetime
import pandas as pd
import numpy as np
import copy

# import swifter


def read_file(path, show_print=False):
    """
    # If reading from a DAT file then use path and the code below
    """
    if show_print:
        print("load file: ", path)
    with open(path, "rb") as binary_file:
        # Read the whole file at once
        data = binary_file.read()
    return data


def read_data(hex_data):
    """read hex string and convert to binary"""
    data = binascii.unhexlify(hex_data[2:])
    return data


def _create_dtype_string(channel, kind):
    DATA_TYPES = {
        "1": "ubyte",
        "2": "byte",
        "3": "ushort",
        "4": "int",
        "5": "ulong",
        "6": "long",
        "7": "float",
        "8": "double",
        "11": "bit16",
        "12": "bit48",
    }
    try:
        dtype = DATA_TYPES[channel["decode_" + kind]]
        if dtype != "double" and dtype != "long" and dtype != "byte":
            dtype += str((int(channel["BytePerValue_" + kind]) * 8))
    except KeyError:
        raise ValueError("Unknown binary format")
    return dtype


def _iter_blocks(data):
    start = 0
    re_block = re.compile(rb"\s*T*\|(\S\S)\s*,\s*(\d+)\s*,\s*(\d+)\s*,")
    while start < len(data):
        entry = re_block.match(data, start)
        if entry is None:
            entry = re_block.match(data, start + 3)
            if entry is None:
                raise ValueError("Block expected, no famos format")
            else:
                start += 3
        typ, num, length = entry.groups()
        length = int(length)
        start = entry.end()
        block = data[start : start + length]
        blockValue = entry[3]
        if start + length == len(data):
            length -= 1
        elif data[start + length] != 59:
            raise ValueError(f"';' expected, found {data[start+length]}")
        start += length + 1
        yield typ, num, block, blockValue


def load_channels(file_or_data, filter=None, show_print=False):
    """Reads in binary file(DAT) or hex string and returns a dictionary
    from extracting information from the input"""
    if isinstance(file_or_data, float):
        return {}
    if file_or_data[:2] == "0x":
        data = read_data(file_or_data)
    elif type(file_or_data) == bytes:
        data = file_or_data
    else:
        data = read_file(file_or_data, show_print=False)
    channels = {}
    channel = None
    count = -1
    for typ, num, block, blockValue in _iter_blocks(data):
        # save information of channel which is for all the same (until******)
        if typ == b"CF":
            pass
        elif typ == b"NL":
            pass
        elif typ == b"CK":
            pass
        elif typ == b"NO":
            pass
        # *********************************************************************
        elif typ == b"CG":
            channel = {}
            channels[count] = channel
            count += 1
        elif typ == b"CD":
            data = block.split(b",")
            # bug, by filteredChannels in SK evaluation as 1s
            channel["delta_x"] = float(data[0])
        elif typ == b"NT":
            data = block.split(b",")
            dd, mm, yy, hour, mins, sec = (
                d.decode("windows-1252").replace(" ", "0") for d in data
            )
            if int(yy) > 2000:  # two locations of trigger time (Cb or NT)
                channel["trigger_time"] = f"{yy}-{mm}-{dd} {hour}:{mins}:{sec}"
            else:
                pass
        elif typ == b"CC":
            pass
        elif typ == b"CP":
            data = block.split(b",")
            try:
                if isinstance(channel["decode_value_y"], str):
                    channel["decode_value_x"] = data[2].decode("windows-1252")
                    channel["BytePerValue_value_x"] = data[1].decode("windows-1252")
            except Exception:
                channel["decode_value_y"] = data[2].decode("windows-1252")
                channel["BytePerValue_value_y"] = data[1].decode("windows-1252")
        elif typ == b"Cb":
            data = block.split(b",")
            try:
                if isinstance(channel["buffer_size_value_y"], int):
                    channel["buffer_size_value_x"] = int(data[5].decode("windows-1252"))
                    channel["buffer_start_value_x"] = int(
                        data[4].decode("windows-1252")
                    )
            except Exception:
                channel["buffer_size_value_y"] = int(data[5].decode("windows-1252"))
                channel["buffer_start_value_y"] = int(data[4].decode("windows-1252"))
            channel["X0"] = float(data[9])
            # then NT block have not the information of triggertime
            if float(data[10]) > 0:
                # famos start 1980 but datetime 1970
                triggertime = (
                    float(data[10])
                    + (datetime.datetime(1980, 1, 1) - datetime.datetime(1970, 1, 1))
                    // datetime.timedelta(seconds=1)
                    - 3600
                )
                try:
                    triggertime = datetime.datetime.fromtimestamp(triggertime).strftime(
                        "%Y-%m-%d %H:%M:%S.%f"
                    )
                except Exception:
                    triggertime = datetime.datetime.fromtimestamp(triggertime).strftime(
                        "%Y-%m-%d %H:%M:%S"
                    )
                channel["trigger_time"] = triggertime
            else:
                pass  # NT block has the information of trigger time
        elif typ == b"ND":
            pass
        elif typ == b"CN":
            data = block.split(b",")
            channel["name"] = data[4].decode("windows-1252")
        elif typ == b"CC":
            pass
        elif typ == b"CR":
            key = "unit_y" if "unit_y" not in channel else "unit_x"
            data = block.split(b",", 6)
            channel[key] = data[5].decode("windows-1252")
            if "°" in channel[key]:
                channel[key] = channel[key].replace("°", "grad")
            elif "²" in channel[key]:
                channel[key] = channel[key].replace("²", "2")
                # channel[key] = channel[key].replace("Â","")
            elif "W/m" in channel[key]:
                channel[key] = channel[key].replace("W/m", "W/m2")
                # channel[key] = channel[key].replace("Â","")
            if "Â" in channel[key]:
                channel[key] = channel[key].replace("Â", "")
            key += "_scale-factor"
            channel[key] = float(data[1].decode("windows-1252"))
            key += "_Offset"
            channel[key] = float(data[2].decode("windows-1252"))
        elif typ == b"ND":
            pass
        elif typ == b"CS":
            _, data = block.split(b",", 1)
            break
    for i in channels:
        scale = channels[i]["unit_y_scale-factor"] or 1.0
        scale_offset = channels[i]["unit_y_scale-factor_Offset"] or 0.0
        if "unit_x" in channels[i]:
            dtype = _create_dtype_string(channels[i], "value_x")
            scale = channels[i]["unit_x_scale-factor"] or 1.0
            scale_offset = channels[i]["unit_x_scale-factor_Offset"] or 0.0
            if (
                channels[i]["unit_x_scale-factor"] != 1
                or channels[i]["unit_x_scale-factor_Offset"] != 0
            ):
                offset = channels[i]["buffer_start_value_x"]
                count = channels[i]["buffer_size_value_x"] // np.dtype(dtype).itemsize
                try:
                    channels[i]["value_x"] = (
                        np.frombuffer(data, dtype=dtype, count=count, offset=offset)
                        * scale
                        + scale_offset
                    )
                except ValueError:
                    print("oops")
            else:
                offset = channels[i]["buffer_start_value_x"]
                count = channels[i]["buffer_size_value_x"] // np.dtype(dtype).itemsize
                channels[i]["value_x"] = np.frombuffer(
                    data, dtype=dtype, count=count, offset=offset
                )
            offset -= count * np.dtype(dtype).itemsize - offset
            scale = channels[i]["unit_y_scale-factor"] or 1.0
        dtype = _create_dtype_string(channels[i], "value_y")
        if (
            channels[i]["unit_y_scale-factor"] != 1
            or channels[i]["unit_y_scale-factor_Offset"] != 0
        ):
            offset = channels[i]["buffer_start_value_y"]
            count = channels[i]["buffer_size_value_y"] // np.dtype(dtype).itemsize
            try:
                channels[i]["value_y"] = (
                    np.frombuffer(data, dtype=dtype, count=count, offset=offset) * scale
                    + scale_offset
                )
            except ValueError:
                raise ValueError("oops")
        else:
            offset = channels[i]["buffer_start_value_y"]
            count = channels[i]["buffer_size_value_y"] // np.dtype(dtype).itemsize
            channels[i]["value_y"] = np.frombuffer(
                data, dtype=dtype, count=count, offset=offset
            )
    # name the indices to channel names#
    channelsOutput = {}
    for i in channels:
        try:
            channelsOutput[channels[i]["name"]] = channels[i]
        except ValueError:
            print("Channel not found: ", i)
    return channelsOutput


def load_spectrum_df(dict_channels):
    list_df = []
    for key, value_dict in dict_channels.items():
        n_rows = len(value_dict["value_y"])
        x_values = np.array(value_dict["delta_x"]).dot(np.arange(n_rows))
        list_df.append(
            pd.DataFrame({"frequency": x_values, key: value_dict["value_y"]})
        )
    if len(list_df) == 1:
        return list_df[0]
    return list_df

def load_series_spectrum_df(series_dict_channels):
    """
    Takes a series of dictionaries generated by pd.Series.apply(load_channels)
    and returns a dataframe with the frequencies expanded as columns.
    If the frequencies are not identically overlapping across rows, the resulting 
    set of columns will the the union of all the different frequency sets, where 
    rows not containing a given frequency will be NaN
    """
    dict_df = {}
    for i, dict_channels in series_dict_channels.items():
        if dict_channels:
            for key, value_dict in dict_channels.items():
                n_rows = len(value_dict['value_y'])
                x_values = np.array(value_dict['delta_x']).dot(np.arange(n_rows))
                for j, freq in enumerate(x_values):
                    try: 
                        dict_df[freq][i] = value_dict['value_y'][j]
                    except KeyError:
                        dict_df[freq] = {i: value_dict['value_y'][j]}
        else:
            pass
    return pd.DataFrame.from_dict(dict_df)

def try_load_channels(x):
    try:
        return load_channels(x)
    except ValueError:
        return None