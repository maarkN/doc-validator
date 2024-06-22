from dataclasses import dataclass
import os
from models.models import DownloadedAudioMetadata
from utils.audiofile import AudioFile
from utils.utils import Timer
from typing import Any
import requests

@dataclass
class FileDownloader:
    @Timer.timer
    @staticmethod
    def _download_timed(url: str, download_path: str):
        if not os.path.exists(os.path.dirname(download_path)):
            os.makedirs(os.path.dirname(download_path), exist_ok=True)
        response = requests.get(url)
        response.raise_for_status()
        with open(download_path, 'wb') as file:
            file.write(response.content)
        return download_path
    
    @staticmethod
    def download(url: str, download_path: str) -> DownloadedAudioMetadata:
        download_path, process_time = FileDownloader._download_timed(url, download_path)
        return DownloadedAudioMetadata(
            audiofile=AudioFile(download_path),
            url=url,
            process_time=process_time
        )
