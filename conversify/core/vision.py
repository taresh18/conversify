import asyncio
import logging
from typing import Dict, Any, Optional, Tuple

from livekit import rtc
from livekit.agents import JobContext

logger = logging.getLogger(__name__)


async def find_video_track(ctx: JobContext) -> Optional[rtc.RemoteVideoTrack]:
    """
    Find and subscribe to the first available video track in the room.
    
    Args:
        ctx: The job context containing room information
        
    Returns:
        The first available RemoteVideoTrack or None if no track is found
    """
    for participant in ctx.room.remote_participants.values():
        if not participant or not participant.track_publications:
            continue
            
        for track_pub in participant.track_publications.values():
            if not track_pub or track_pub.kind != rtc.TrackKind.KIND_VIDEO:
                continue
                
            # Attempt to subscribe if not already subscribed
            if not track_pub.subscribed:
                logger.info(f"Subscribing to video track {track_pub.sid} from {participant.identity}...")
                try:
                    track_pub.set_subscribed(True)
                    # Wait for subscription to complete
                    await asyncio.sleep(0.5)
                except Exception as e:
                    logger.warning(f"Failed to subscribe to track {track_pub.sid}: {e}")
                    continue
            
            # Check if track is available after subscription
            if (track_pub.track and 
                isinstance(track_pub.track, rtc.RemoteVideoTrack) and 
                track_pub.subscribed):
                logger.info(f"Found video track: {track_pub.track.sid} from {participant.identity}")
                return track_pub.track
    
    return None


async def video_processing_loop(ctx: JobContext, shared_state: Dict[str, Any], video_frame_interval: float) -> None:
    """
    Process the first available video track and update shared_state['latest_image'].
    
    Args:
        ctx: The job context containing room information
        shared_state: Dictionary to store shared data between components
        video_frame_interval: Interval (seconds) between frame processing
    """
    if not ctx:
        logger.error("Invalid arguments: JobContext is None")
        return

    logger.info("Starting video processing loop, looking for video track...")
    video_track = None
    video_stream = None
    
    try:
        while True:
            try:
                video_track = await find_video_track(ctx)
                if video_track:
                    break
                    
                logger.debug("No video track found yet, waiting...")
                await asyncio.sleep(1)
            except asyncio.CancelledError:
                logger.info("Video track search cancelled.")
                return
            except Exception as e:
                logger.error(f"Error searching for video track: {e}", exc_info=True)
                await asyncio.sleep(1)  # Wait before retrying

        # Create video stream from the found track
        video_stream = rtc.VideoStream(video_track)
        logger.info(f"Starting video stream processing with interval {video_frame_interval}s.")
        
        # Process the video stream
        async for event in video_stream:
            if event and event.frame:
                # Update the shared state with the latest frame
                shared_state['latest_image'] = event.frame
                
            # Sleep to control processing rate
            await asyncio.sleep(video_frame_interval)
            
    except asyncio.CancelledError:
        logger.info("Video processing task cancelled.")
    except Exception as e:
        logger.error(f"Error processing video stream: {e}", exc_info=True)
    finally:
        # Clean up resources
        if video_stream:
            try:
                await video_stream.aclose()
                logger.info("Video stream closed.")
            except Exception as e:
                logger.error(f"Error closing video stream: {e}")
        
        # Clear the latest image reference when processing ends
        shared_state.pop('latest_image', None)
        logger.info("Video processing loop ended.") 