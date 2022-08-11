import tqdm
import argparse
from pathlib import Path
from enum import Enum

import numpy as np
import logging

import PyNvCodec as nvc

logger = logging.getLogger(__file__)


class DecodeStatus(Enum):
    # Decoding error.
    DEC_ERR = (0,)
    # Frame was submitted to decoder.
    # No frames are ready for display yet.
    DEC_SUBM = (1,)
    # Frame was submitted to decoder.
    # There's a frame ready for display.
    DEC_READY = 2


class NvTranscode:
    def __init__(
        self,
        gpu_id: int,
        enc_file: str,
        dec_file: str,
    ):
        self.nv_dmx = None
        self.nv_dec = nvc.PyNvDecoder(enc_file, gpu_id)

        # Frame to seek to next time decoding function is called.
        # Negative values means 'don't use seek'.  Non-negative values mean
        # seek frame number.
        self.sk_frm = int(-1)
        # Total amount of decoded frames
        self.num_frames_decoded = int(0)
        # Numpy array to store decoded frames pixels
        self.frame_nv12 = np.ndarray(shape=(0), dtype=np.uint8)
        # Output file
        self.out_file = open(dec_file, "wb")

        (width, height) = (self.nv_dec.Width(), self.nv_dec.Height())

        res = str(width) + "x" + str(height)

        self.nvEnc = nvc.PyNvEncoder(
            {
                "preset": "P1",
                "tuning_info": "low_latency",
                "codec": "hevc",
                "profile": "main",
                "s": res,
                "bitrate": "4M",
                "rc": "vbr",
                "gop": "30",
            },
            gpu_id,
        )

        self.frame_nv12_enc = np.ndarray(shape=(0), dtype=np.uint8)

    def decode_frame(self, verbose=False) -> DecodeStatus:
        status = DecodeStatus.DEC_ERR
        try:
            self.frame_nv12 = self.nv_dec.DecodeSingleSurface()
            status = DecodeStatus.DEC_READY
        except Exception as e:
            logger.info(f"{getattr(e, 'message', str(e))}")

        return status

    # Write current video frame to output file.
    def encode_frame(self) -> None:
        success = self.nvEnc.EncodeSingleSurface(self.frame_nv12, self.frame_nv12_enc)
        if success:
            byte_me = bytearray(self.frame_nv12_enc)
            self.out_file.write(byte_me)
            self.num_frames_decoded += 1

    # Decode all available video frames and write them to output file.
    def transcode(self, frames_to_decode=-1, verbose=False) -> None:
        # Main decoding cycle
        pbar = tqdm.tqdm(range(self.nv_dec.Numframes()), ascii=True, unit=" frames")
        pbar.set_description("Decoding -> Encoding ")
        for index in pbar:
            status = self.decode_frame(verbose)
            if status == DecodeStatus.DEC_ERR:
                break
            elif status == DecodeStatus.DEC_READY:
                self.encode_frame()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        "This sample decodes src video (--src) to RGB frames and then encodes them back to file (--dst) on given GPU."
    )
    parser.add_argument(
        "-g",
        "--gpu-id",
        type=int,
        required=True,
        help="GPU id, check nvidia-smi",
    )
    parser.add_argument(
        "-s",
        "--src",
        type=Path,
        required=True,
        help="Source file path",
    )
    parser.add_argument(
        "-d",
        "--dst",
        type=Path,
        required=True,
        help="Destination file path",
    )

    args = parser.parse_args()

    tc = NvTranscode(
        args.gpu_id,
        args.src.as_posix(),
        args.dst.as_posix(),
    )
    tc.transcode()

    exit(0)
