"""
Video Processor — Upload handling, transcoding, and segmentation.
Production pipeline for tennis match video processing.
"""

from __future__ import annotations
import os
import uuid
import asyncio
from enum import Enum
from dataclasses import dataclass, field
from typing import Optional
from datetime import datetime


class VideoStatus(str, Enum):
    UPLOADED = "uploaded"
    QUEUED = "queued"
    TRANSCODING = "transcoding"
    SEGMENTING = "segmenting"
    ANALYZING = "analyzing"
    COMPLETE = "complete"
    FAILED = "failed"


class VideoCodec(str, Enum):
    H264 = "h264"
    H265 = "h265"
    VP9 = "vp9"
    AV1 = "av1"


class VideoResolution(str, Enum):
    SD_480 = "480p"
    HD_720 = "720p"
    FHD_1080 = "1080p"
    UHD_4K = "2160p"


@dataclass
class TranscodeProfile:
    """Encoding profile for a specific quality tier."""
    name: str
    resolution: VideoResolution
    bitrate_kbps: int
    codec: VideoCodec
    fps: int = 30
    keyframe_interval: int = 60  # 2 seconds at 30fps

    @property
    def ffmpeg_args(self) -> list[str]:
        res_map = {"480p": "854:480", "720p": "1280:720", "1080p": "1920:1080", "2160p": "3840:2160"}
        codec_map = {"h264": "libx264", "h265": "libx265", "vp9": "libvpx-vp9", "av1": "libaom-av1"}
        return [
            "-vf", f"scale={res_map[self.resolution.value]}",
            "-c:v", codec_map[self.codec.value],
            "-b:v", f"{self.bitrate_kbps}k",
            "-r", str(self.fps),
            "-g", str(self.keyframe_interval),
            "-preset", "fast",
        ]


# Standard ABR profiles for HLS streaming
TRANSCODE_PROFILES = [
    TranscodeProfile("low", VideoResolution.SD_480, 800, VideoCodec.H264),
    TranscodeProfile("medium", VideoResolution.HD_720, 2500, VideoCodec.H264),
    TranscodeProfile("high", VideoResolution.FHD_1080, 5000, VideoCodec.H264),
]


@dataclass
class VideoSegment:
    """A segment/point extracted from a match video."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    video_id: str = ""
    segment_type: str = "point"   # point, game, set, highlight
    start_time_ms: int = 0
    end_time_ms: int = 0
    point_number: Optional[int] = None
    score_at_start: str = ""
    score_at_end: str = ""
    tags: list[str] = field(default_factory=list)


@dataclass
class VideoJob:
    """Represents a video processing job."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    session_id: str = ""
    source_path: str = ""
    output_dir: str = ""
    status: VideoStatus = VideoStatus.UPLOADED
    progress: float = 0.0
    created_at: datetime = field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    duration_seconds: float = 0.0
    fps: float = 30.0
    width: int = 1920
    height: int = 1080
    segments: list[VideoSegment] = field(default_factory=list)
    transcode_outputs: dict[str, str] = field(default_factory=dict)


class VideoProcessor:
    """
    Production video processing pipeline.

    Handles:
    1. Upload & validation
    2. Transcoding to multiple bitrates (ABR for HLS)
    3. Point-level segmentation from scoring timeline
    4. Frame extraction for ML inference
    """

    def __init__(self, upload_dir: str = "/tmp/tennisiq/uploads", output_dir: str = "/tmp/tennisiq/output"):
        self.upload_dir = upload_dir
        self.output_dir = output_dir
        self._jobs: dict[str, VideoJob] = {}
        os.makedirs(upload_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)

    def create_job(self, session_id: str, source_path: str) -> VideoJob:
        """Create a new video processing job."""
        job_output = os.path.join(self.output_dir, session_id)
        os.makedirs(job_output, exist_ok=True)
        job = VideoJob(session_id=session_id, source_path=source_path, output_dir=job_output)
        self._jobs[job.id] = job
        return job

    async def process(self, job_id: str) -> VideoJob:
        """Run the full video processing pipeline."""
        job = self._jobs.get(job_id)
        if not job:
            raise ValueError(f"Job {job_id} not found")

        try:
            # Step 1: Probe video metadata
            job.status = VideoStatus.QUEUED
            job.progress = 0.05
            await self._probe_video(job)

            # Step 2: Transcode to ABR profiles
            job.status = VideoStatus.TRANSCODING
            job.progress = 0.1
            await self._transcode(job)

            # Step 3: Segment by points
            job.status = VideoStatus.SEGMENTING
            job.progress = 0.6
            await self._segment_by_points(job)

            # Step 4: Generate HLS manifest
            job.progress = 0.9
            await self._generate_hls_manifest(job)

            # Done
            job.status = VideoStatus.COMPLETE
            job.progress = 1.0
            job.completed_at = datetime.utcnow()

        except Exception as e:
            job.status = VideoStatus.FAILED
            job.error_message = str(e)

        return job

    async def _probe_video(self, job: VideoJob):
        """Extract video metadata using ffprobe."""
        # In production: run ffprobe subprocess
        # Simulated for scaffolding
        job.duration_seconds = 7200.0  # 2 hour match
        job.fps = 30.0
        job.width = 1920
        job.height = 1080

    async def _transcode(self, job: VideoJob):
        """Transcode video to multiple quality levels."""
        for i, profile in enumerate(TRANSCODE_PROFILES):
            output_path = os.path.join(job.output_dir, f"{profile.name}.mp4")
            # In production: run ffmpeg subprocess
            # ffmpeg_cmd = ["ffmpeg", "-i", job.source_path] + profile.ffmpeg_args + [output_path]
            job.transcode_outputs[profile.name] = output_path
            job.progress = 0.1 + (0.5 * (i + 1) / len(TRANSCODE_PROFILES))
            await asyncio.sleep(0)  # yield control

    async def _segment_by_points(self, job: VideoJob):
        """Segment video into individual points using scoring timeline."""
        # In production: use scoring timeline timestamps to split video
        # Example segments
        for i in range(10):
            seg = VideoSegment(
                video_id=job.id,
                segment_type="point",
                start_time_ms=i * 30000,
                end_time_ms=(i + 1) * 30000 - 2000,
                point_number=i + 1,
                tags=["rally"],
            )
            job.segments.append(seg)

    async def _generate_hls_manifest(self, job: VideoJob):
        """Generate HLS master playlist and media playlists."""
        manifest_path = os.path.join(job.output_dir, "master.m3u8")
        manifest = "#EXTM3U\n#EXT-X-VERSION:3\n"
        bandwidth_map = {"low": 800000, "medium": 2500000, "high": 5000000}
        res_map = {"low": "854x480", "medium": "1280x720", "high": "1920x1080"}
        for name, path in job.transcode_outputs.items():
            bw = bandwidth_map.get(name, 2500000)
            res = res_map.get(name, "1280x720")
            manifest += f'#EXT-X-STREAM-INF:BANDWIDTH={bw},RESOLUTION={res}\n{name}.m3u8\n'
        # In production: write to file system / S3
        job.transcode_outputs["manifest"] = manifest_path

    def get_job(self, job_id: str) -> Optional[VideoJob]:
        return self._jobs.get(job_id)

    def extract_frames(self, job: VideoJob, start_ms: int, end_ms: int, fps: int = 5) -> list[str]:
        """Extract frames from a video segment for ML inference."""
        # In production: ffmpeg frame extraction
        frame_count = int((end_ms - start_ms) / 1000 * fps)
        return [f"frame_{i:06d}.jpg" for i in range(frame_count)]
