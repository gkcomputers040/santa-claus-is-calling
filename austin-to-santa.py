###########
###########
# ==========================================================================================================
# SANTA CLAUS AI - INTELLIGENT PHONE CALL SYSTEM
# ==========================================================================================================
#
# APPLICATION FLOW OVERVIEW:
# --------------------------
# This application creates an AI-powered phone call system where Santa Claus calls children.
# The flow works as follows:
#
# 1. CALL INITIATION:
#    - A call is scheduled through the /schedule-call endpoint or initiated via caller.py
#    - When the call is answered, Twilio triggers the /answer endpoint
#
# 2. STREAM CONNECTION:
#    - The /answer endpoint redirects the call to a WebSocket stream connection
#    - This allows real-time bidirectional audio communication
#
# 3. AUDIO PROCESSING:
#    - The stream() WebSocket function receives incoming audio from the caller
#    - Audio is sent to Deepgram for real-time speech-to-text transcription
#
# 4. TRANSCRIPTION HANDLING:
#    - When Deepgram transcribes speech, get_transcription_add_call_sid() is triggered
#    - This function receives the transcription, interprets it, and sends it to the LLM (GPT/Claude)
#
# 5. AI RESPONSE GENERATION:
#    - The LLM receives the transcription along with the conversation history
#    - It generates a response based on Santa's persona and the child's information
#
# 6. TEXT-TO-SPEECH CONVERSION:
#    - The LLM response is split into chunks for faster processing
#    - Each chunk is sent to ElevenLabs TTS for high-quality voice synthesis
#
# 7. AUDIO PLAYBACK:
#    - The generated audio is converted to Twilio's required format (mulaw, 8000Hz)
#    - Audio is streamed back to the caller through the WebSocket connection
#
# 8. CONVERSATION LOOP:
#    - Steps 3-7 repeat continuously until the call ends or time runs out
#
# ==========================================================================================================
###########

# ==========================================================================================================
# LIBRARY IMPORTS
# ==========================================================================================================
# We import all necessary libraries at the top of the file following Python best practices.
# Each library serves a specific purpose in the application:

# Standard library imports for basic functionality
import io                   # For handling byte streams (audio data manipulation)
import os                   # For accessing environment variables and file system operations
import re                   # For regular expressions (text pattern matching and manipulation)
import re                   # Duplicate import (can be removed, but kept for compatibility)
import subprocess           # For executing external scripts (caller.py)
import sys                  # For system-specific parameters and functions
import json                 # For parsing and creating JSON data structures
import time                 # For time-related operations (timestamps, delays)
import uuid                 # For generating unique identifiers (job IDs)
import wave                 # For handling WAV audio files
import pydub                # For audio format conversion and manipulation
import emoji                # For handling emoji characters in text
import orjson               # High-performance JSON library (faster than standard json)
import atexit               # For registering cleanup functions to run at program exit
import base64               # For encoding/decoding audio data to Base64 (Twilio requirement)
import openai               # OpenAI API client for GPT models
import random               # For generating random numbers (unique IDs, random selections)
import psutil               # For accessing system information (CPU count for connection pooling)
import string               # For string constants (letters, digits, punctuation)
import aiohttp              # Async HTTP client for making API requests
import asyncio              # For asynchronous programming (handling concurrent operations)
import audioop              # For audio operations (converting to mulaw format)
import logging              # For application logging (debugging and monitoring)
import sqlite3              # For SQLite database operations (user data storage)
import aiofiles             # For async file operations (loading audio files)
import requests             # For synchronous HTTP requests
import tempfile             # For creating temporary files
import threading            # For multi-threading (running HTTP and HTTPS servers)
import pytz, sqlite3, uuid  # Additional imports for timezone handling and database
import numpy as np          # For numerical operations (potential audio processing)
from io import BytesIO      # For creating byte stream objects
from queue import Queue     # For queue data structures
from asyncio import sleep   # For async sleep operations
from pytz import timezone   # For timezone conversions
from scipy.io import wavfile # For reading/writing WAV files
from datetime import datetime # For date and time operations
from deepgram import Deepgram # Deepgram SDK for speech-to-text
from dotenv import load_dotenv # For loading environment variables from .env file
from pydub import AudioSegment # For audio manipulation (format conversion)
from pydantic import BaseModel # For data validation and settings management
from twilio.rest import Client # Twilio SDK for making and managing phone calls
from aiohttp import ClientSession # Async HTTP session for persistent connections
from multiprocessing import Value # For shared memory between processes
from urllib.parse import urlencode # For URL encoding query parameters
from starlette.routing import Route # For defining routes in FastAPI
from queue import PriorityQueue, Queue # For priority queue data structures
from starlette.responses import Response # For creating HTTP responses
from fastapi.staticfiles import StaticFiles # For serving static files
from concurrent.futures import ThreadPoolExecutor # For thread pool execution
from twilio.base.exceptions import TwilioRestException # For handling Twilio API errors
from websockets.exceptions import ConnectionClosedError # For handling WebSocket errors
from apscheduler.schedulers.background import BackgroundScheduler # For scheduling calls
from twilio.twiml.voice_response import Connect, VoiceResponse, Stream # For Twilio TwiML responses
from fastapi.responses import JSONResponse, FileResponse, StreamingResponse, HTMLResponse # HTTP response types
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect, Depends, HTTPException, status # FastAPI core

from asyncio import ensure_future  # For scheduling coroutines

# ==========================================================================================================
# APPLICATION INITIALIZATION
# ==========================================================================================================
# Initialize the FastAPI application instance.
# FastAPI is a modern, high-performance web framework for building APIs with Python.
# It provides automatic API documentation, request validation, and async support.
app = FastAPI()

# ==========================================================================================================
# SCHEDULER INITIALIZATION
# ==========================================================================================================
# Initialize the APScheduler BackgroundScheduler for scheduling phone calls.
# The scheduler runs in the background and triggers calls at specified times.
# This allows users to schedule calls for specific dates and times.
scheduler = BackgroundScheduler()
scheduler.start()

# ==========================================================================================================
# ENVIRONMENT CONFIGURATION
# ==========================================================================================================
# Load environment variables from the .env file.
# This is a secure way to store sensitive information like API keys and credentials
# without hardcoding them in the source code. The .env file should never be committed
# to version control to protect sensitive data.
load_dotenv()

# Get the database filename from environment variables.
# This allows different environments (development, production) to use different databases.
dbname = os.getenv("DATABASE")

# ==========================================================================================================
# CONNECTION POOL CONFIGURATION
# ==========================================================================================================
# Calculate the optimal connection pool size based on CPU count.
# This follows the common formula: pool_size = (num_cpus * 2) + 1
# This ensures efficient handling of concurrent connections without overwhelming the system.
# The formula accounts for both CPU-bound and I/O-bound operations.
num_cpus = psutil.cpu_count(logical=False)  # Get physical CPU count (not logical cores)
max_pool = num_cpus * 2 + 1  # Calculate optimal pool size

# ==========================================================================================================
# GOOGLE CREDENTIALS INITIALIZATION
# ==========================================================================================================
# Set the Google Application Credentials environment variable.
# This is required for Google Cloud services authentication (if used).
# The credentials file contains service account information for API access.
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'g_credentials.json'

# ==========================================================================================================
# LLM (LARGE LANGUAGE MODEL) CONFIGURATION
# ==========================================================================================================
# Determine which AI model to use for generating responses.
# The application supports both OpenAI's GPT models and Anthropic's Claude models.
# The choice is made via environment variable to allow easy switching between providers.
llm_ai = os.getenv("LLM_AI")
if llm_ai == "Claude":
    # Use Anthropic's Claude model
    model_ai = os.getenv("MODEL_CLAUDE")
    rol = "santa-calling_1_Claude"  # Claude-specific role/prompt file
else:
    # Default to OpenAI's GPT model
    model_ai = os.getenv("MODEL_GPT")
    rol = "santa-calling_1"  # GPT-specific role/prompt file

print(f"model_ai: {model_ai}")

# ==========================================================================================================
# API CREDENTIALS INITIALIZATION
# ==========================================================================================================
# Load all API keys and credentials from environment variables.
# These are essential for authenticating with external services.

# OpenAI API key for GPT models
openai_key = os.getenv("OPENAI_KEY")

# Anthropic API key for Claude models
claude_key = os.getenv('ANTHROPIC_API_KEY')

# Twilio credentials for phone call management
# account_sid: Unique identifier for your Twilio account
# auth_token: Secret token for API authentication
account_sid = os.getenv("TWILIO_SID")
auth_token = os.getenv("TWILIO_AUTH")

# ElevenLabs API key for high-quality text-to-speech conversion
elevenlabs_key = os.getenv("ELEVEN_KEY")

# Deepgram API key for real-time speech-to-text transcription
deepgram_key = os.getenv("DEEPGRAM_KEY")

# WebSocket URL for Twilio stream connections
# This is the public URL where Twilio will connect for audio streaming
websockets_url = os.getenv("WEBSOCKET_URL")

# URL for the intro audio that plays when call is answered
intro_audio_url = os.getenv("INTRO_AUDIO_URL")

# Set OpenAI API key globally for the openai library
openai.api_key = openai_key

# Initialize Twilio client with account credentials
# This client is used for all Twilio API operations (making calls, updating calls, etc.)
twilio_client = Client(account_sid, auth_token)

# ==========================================================================================================
# PYDANTIC MODEL FOR REQUEST VALIDATION
# ==========================================================================================================
# Define a Pydantic model for validating cancel-call request bodies.
# Pydantic provides automatic data validation and type checking.
# This ensures that incoming API requests have the correct data structure.
class CancelCallRequest(BaseModel):
    user_id: int  # The unique identifier of the user whose call should be cancelled

# ==========================================================================================================
# GLOBAL STATE VARIABLES
# ==========================================================================================================
# These variables store application state that needs to be shared across functions.
# Note: In production, consider using a proper state management solution (Redis, database)
# for better scalability and persistence.

# Accumulated transcript - stores partial transcriptions until speech is finalized
accumulated_transcript = ""

# Timestamp of last received transcription - used for timeout handling
last_received_time = None

# ==========================================================================================================
# AUDIO FILE CACHING
# ==========================================================================================================
# Load intro audio files into memory at startup.
# This improves performance by avoiding disk I/O during calls.
# The files are loaded once and reused for all calls.
intro_spanish_mp3 = None  # Spanish intro audio
intro_english_mp3 = None  # English intro audio

# ==========================================================================================================
# ROLE/PROMPT CONFIGURATION
# ==========================================================================================================
# Set the path to the role/prompt file that defines Santa's personality and behavior.
# This file contains instructions for the LLM on how to behave as Santa Claus.
role_file_path = f"roles/{rol}.txt"

# ==========================================================================================================
# CALL STATE DICTIONARIES
# ==========================================================================================================
# These dictionaries store state information for each active call.
# The key is always the call_sid (Twilio's unique call identifier).
# Using call_sid as key allows managing multiple concurrent calls.

# conversations: Stores the message history for each call
# Format: {call_sid: [{"role": "user/assistant/system", "content": "message"}]}
# This is essential for maintaining context in the conversation
conversations = {}

# connector: Stores aiohttp TCP connectors for each call
# These manage connection pooling for API requests
connector = {}

# session: Stores aiohttp client sessions for each call
# Sessions maintain cookies and connection pools for HTTP requests
session = {}

# TTS_Index: Tracks text-to-speech chunk ordering for each call
# Ensures audio chunks are played in the correct sequence
TTS_Index = {}

# TTS_Queue: Queue for pending TTS operations for each call
# Allows buffering of text chunks waiting to be converted to speech
TTS_Queue = {}

# TTS_Audio: Stores generated audio data for each call
# Used for caching and playback management
TTS_Audio = {}

# connected_websockets: Stores active WebSocket connections for each call
# Allows sending audio back to the correct call
connected_websockets = {}

# gpt_arguments: Stores function call arguments from GPT for each call
# Used when GPT requests to execute specific functions
gpt_arguments = {}

# gpt_talking: Boolean flag indicating if the AI is currently speaking
# Prevents processing new transcriptions while AI is responding
# This avoids overlapping responses and confusion
gpt_talking = {}

# deepgram_live: Stores Deepgram live transcription instances for each call
# Each call has its own Deepgram connection for transcription
deepgram_live = {}

# full_transcription: Accumulates complete transcription for each call
# Combines partial transcriptions into full sentences before sending to LLM
full_transcription = {}

# time_transcription: Stores timestamp of last transcription for each call
# Used for implementing timeout logic (send after X seconds of silence)
time_transcription = {}

# call_extra_info: Stores additional call metadata for each call
# Includes child's name, parents' names, gifts, context, language, timer, etc.
call_extra_info = {}


# ==========================================================================================================
# EMOJI CONSTANTS
# ==========================================================================================================
# Define emoji constants used in the application.
# The phone emoji is used as a signal for GPT to hang up the call.
# When GPT includes this emoji in its response, the call will be terminated.
phone_emoji = emoji.emojize(':telephone:')

# Query parameters - stores URL query parameters for the current request
query_params = ""


# ==========================================================================================================
# AUDIO FILE LOADING
# ==========================================================================================================
async def load_mp3_files():
    """
    Loads intro audio files into memory at application startup.

    This function is called during application initialization to preload
    the Spanish and English intro audio files. Loading them into memory
    eliminates disk I/O during calls, reducing latency.

    The intro audio contains Santa's initial greeting ("Ho ho ho!") that
    plays immediately when the call is answered, while the LLM prepares
    its first response.

    Uses aiofiles for async file reading to avoid blocking the event loop.
    """
    global intro_spanish_mp3, intro_english_mp3

    # Load Spanish intro audio asynchronously
    async with aiofiles.open("static/audio/intro-Spanish.mp3", mode='rb') as f:
        intro_spanish_mp3 = await f.read()

    # Load English intro audio asynchronously
    async with aiofiles.open("static/audio/intro-English.mp3", mode='rb') as f:
        intro_english_mp3 = await f.read()


# ==========================================================================================================
# DATABASE CONNECTION
# ==========================================================================================================
def get_db_connection():
    """
    Creates and returns a connection to the SQLite database.

    This function creates a new database connection each time it's called.
    SQLite connections are not thread-safe by default, so each operation
    should use its own connection.

    Returns:
        sqlite3.Connection: A connection object to the SQLite database.

    Note: The caller is responsible for closing the connection after use
    to prevent resource leaks.
    """
    conn = sqlite3.connect(dbname)
    return conn


# ==========================================================================================================
# SECURITY: LOCALHOST CHECK
# ==========================================================================================================
def is_localhost(request: Request) -> bool:
    """
    Checks if a request originates from localhost (127.0.0.1 or ::1).

    This security function is used to restrict certain endpoints to internal
    use only. Endpoints like /schedule-call and /cancel-call should only be
    accessible from the local machine or internal services, not from external
    clients.

    Args:
        request: The FastAPI Request object containing client information.

    Returns:
        bool: True if the request comes from localhost, False otherwise.

    Security Note: This helps prevent unauthorized access to sensitive
    administrative endpoints.
    """
    client_host = request.client.host if request.client else None
    return client_host in ("127.0.0.1", "::1")


# ==========================================================================================================
# DEEPGRAM INITIALIZATION AND TRANSCRIPTION HANDLING
# ==========================================================================================================
async def setup_deepgram_sdk(call_sid, streamSid):
    """
    Initializes Deepgram for real-time speech-to-text transcription.

    This function sets up a live transcription connection with Deepgram's API.
    Deepgram is used instead of other STT services because it provides:
    - Real-time streaming transcription (lower latency)
    - High accuracy for conversational speech
    - Support for multiple languages
    - Interim results (partial transcriptions while speaking)

    The function creates a Deepgram live connection and registers event handlers
    for receiving transcriptions and handling connection close events.

    Args:
        call_sid: Twilio's unique identifier for the current call.
                  Used as a key to store the Deepgram instance.
        streamSid: Twilio's stream identifier for the audio connection.
                   Used when sending audio responses back.

    Returns:
        deepgram_live instance: The initialized Deepgram transcription object,
                                or None if initialization fails.

    Technical Details:
    - smart_format: True - Enables intelligent formatting (numbers, dates)
    - interim_results: True - Provides partial results while speaking
    - language: Set from call_extra_info for the call's language
    - model: nova-2 - Deepgram's latest, most accurate model
    - encoding: mulaw - Twilio's audio encoding format
    - sample_rate: 8000 - Twilio's audio sample rate (8kHz)
    """
    global received_handler, gpt_talking, call_extra_info

    # Initialize Deepgram client with API key
    deepgram = Deepgram(deepgram_key)

    try:
        # Create a live transcription connection with specified parameters.
        # These parameters are optimized for Twilio's audio format and real-time conversation.
        deepgram_live[call_sid] = await deepgram.transcription.live({
            'smart_format': True,      # Enable smart formatting for numbers, dates, etc.
            'interim_results': True,   # Get partial results while user is speaking
            'language': call_extra_info[call_sid]['lang'],  # Language code (en, es, etc.)
            'model': 'nova-2',         # Use Deepgram's latest Nova 2 model
            'encoding': 'mulaw',       # Twilio uses mulaw encoding
            'sample_rate': 8000        # Twilio audio is 8kHz
        })

    except ConnectionClosedError as e:
        # Handle case where the connection closes unexpectedly
        print(f'Connection closed with code {e.code} and reason {e.reason}.')
        return None
    except Exception as e:
        # Handle any other initialization errors
        print(f'Could not open socket: {e}')
        return None

    # Register handler for when the streaming connection closes.
    # This helps with debugging and error tracking.
    deepgram_live[call_sid].registerHandler(
        deepgram_live[call_sid].event.CLOSE,
        handle_closing_event
    )


    # ==========================================================================================================
    # TRANSCRIPTION CALLBACK FUNCTION
    # ==========================================================================================================
    async def get_transcription_add_call_sid(transcription):
        """
        Callback function that processes each transcription received from Deepgram.

        This is the heart of the conversation flow. Every time Deepgram detects
        speech and transcribes it, this function is called. It handles:

        1. Timer management - Updates remaining call time
        2. Transcription accumulation - Combines partial transcriptions
        3. LLM communication - Sends complete sentences to GPT/Claude
        4. Response handling - Processes AI responses

        The function distinguishes between:
        - Interim results: Partial transcriptions while user is still speaking
        - Final results: Complete transcription of a finished utterance
        - Speech final: Complete sentence ready to be processed

        This distinction is crucial for natural conversation flow - we don't want
        to interrupt the AI or send partial sentences to the LLM.

        Args:
            transcription: JSON object from Deepgram containing:
                - channel.alternatives[0].transcript: The transcribed text
                - is_final: Whether this is the final version of this audio segment
                - speech_final: Whether the speaker has finished speaking

        Flow:
        1. Update call timer
        2. Check for transcription content
        3. If speech is final, send to LLM
        4. Process LLM response through TTS
        5. Re-enable transcription listener
        """
        global accumulated_transcript, last_received_time, sentence_transcript
        global mensajes, deepgram_live, received_handler, connected_websockets
        global gpt_talking, full_transcription, time_transcription

        # Generate unique ID for debugging/tracing this specific transcription
        id_unico = random.uniform(1, 100000)

        # ==========================================================================================================
        # TIMER MANAGEMENT
        # ==========================================================================================================
        # Update the call timer. The timer tracks how much time remains for this call.
        # This is important for paid services where calls have time limits.
        update_timer(call_sid, False)

        # Safety measure: Force hang up after 30 minutes (1890 seconds) absolute maximum.
        # This prevents runaway calls that could incur excessive costs.
        time_lapsed = round(time.time() - int(call_extra_info[call_sid]['start_time']))
        if time_lapsed >= 1890:
            await hang_up_call(call_sid)

        # If remaining time is -90 seconds (90 seconds past the limit),
        # force hang up. This gives a 90-second grace period for goodbye.
        if call_extra_info[call_sid]['remaining_time'] <= -90:
            print(f"{call_sid} - TIME IS UP!!")
            await hang_up_call(call_sid)


        mensajes = []

        # Initialize full_transcription for this call if not exists
        if call_sid not in full_transcription:
            full_transcription[call_sid] = ""

        # ==========================================================================================================
        # EXTRACT TRANSCRIPTION TEXT
        # ==========================================================================================================
        # Check if the transcription contains the "channel" key.
        # Deepgram may send other types of messages that don't contain transcriptions.
        if "channel" in transcription:
            sentence_transcript = transcription["channel"]["alternatives"][0]["transcript"]
        else:
            sentence_transcript = None

        # Only process if there's actual transcribed text
        if (sentence_transcript):

            # ==========================================================================================================
            # HANDLE FINAL TRANSCRIPTIONS
            # ==========================================================================================================
            if transcription["is_final"]:

                # ==========================================================================================================
                # TIMEOUT HANDLING FOR PARTIAL TRANSCRIPTIONS
                # ==========================================================================================================
                # Sometimes Deepgram detects speech but doesn't mark it as finalized.
                # If more than 5 seconds pass since the last transcription, force it to be final.
                # This prevents getting stuck waiting for a "speech_final" that never comes.
                if call_sid in time_transcription:
                    transcription_elapsed_time = time.time() - time_transcription[call_sid]
                    if round(transcription_elapsed_time) >= 5:
                        del time_transcription[call_sid]
                        transcription["speech_final"] = True

                # ==========================================================================================================
                # PROCESS COMPLETE UTTERANCES
                # ==========================================================================================================
                if transcription["speech_final"]:

                    # Clear the timeout timer since we received final speech
                    if call_sid in time_transcription:
                        del time_transcription[call_sid]

                    # ==========================================================================================================
                    # TEMPORARILY DISABLE TRANSCRIPTION LISTENER
                    # ==========================================================================================================
                    # Unregister the transcription handler while processing.
                    # This prevents new transcriptions from being processed while the AI is
                    # generating and speaking its response. Without this, overlapping
                    # conversations would create confusion.
                    deepgram_live[call_sid].deregister_handler(
                        deepgram_live[call_sid].event.TRANSCRIPT_RECEIVED,
                        get_transcription_add_call_sid
                    )

                    # Only process if Santa is not currently talking
                    if gpt_talking[call_sid] == False:

                        # ==========================================================================================================
                        # ACCUMULATE TRANSCRIPTION
                        # ==========================================================================================================
                        # Add the new transcription to the accumulated text.
                        # Add space before new text if there's existing content.
                        if full_transcription[call_sid]:
                            full_transcription[call_sid] += " "
                        full_transcription[call_sid] += sentence_transcript

                        # ==========================================================================================================
                        # TIME WARNING SIGNALS
                        # ==========================================================================================================
                        # Add special emoji signals to tell the LLM about time constraints.
                        # These are defined in the role prompt so the LLM knows how to interpret them.

                        # ⌛️ (hourglass) - 90 seconds or less remaining, keep responses shorter
                        if call_extra_info[call_sid]['remaining_time'] <= 90:
                            print("90 seconds remaining!")
                            full_transcription[call_sid] += "⌛️"
                        # ⏰️ (alarm clock) - 30 seconds or less remaining, start saying goodbye
                        elif call_extra_info[call_sid]['remaining_time'] <= 30:
                            print("TIME IS UP!")
                            full_transcription[call_sid] += "⏰️"

                        print(f"{call_sid} - FINAL TRANSCRIPTION: {full_transcription[call_sid]}")

                        # ==========================================================================================================
                        # PREPARE AND SEND TO LLM
                        # ==========================================================================================================
                        # Get the message history for this call (all previous exchanges)
                        message_history = conversations[call_sid]

                        # Create the user message object with the transcription
                        user_message = {"role": "user", "content": full_transcription[call_sid]}

                        # Add to message history before sending to LLM.
                        # The LLM needs the complete conversation context to generate
                        # appropriate responses.
                        message_history.append(user_message)

                        # ==========================================================================================================
                        # GET LLM RESPONSE
                        # ==========================================================================================================
                        # Send the message history to the LLM and get the response.
                        # The response will be streamed, converted to speech, and sent to Twilio.
                        full_content, message_history = await send_msg_gpt(
                            session, call_sid, message_history, streamSid, llm_ai
                        )

                        # Clear the transcription buffer after processing
                        full_transcription[call_sid] = ""
                    else:
                        # Log when user speaks while Santa is talking.
                        # Currently, this doesn't interrupt Santa, but could be implemented
                        # for more natural conversation flow.
                        print(f"{call_sid} - user talking while Santa talks (currently does not stop even if talking): {sentence_transcript}")

                    # ==========================================================================================================
                    # RE-ENABLE TRANSCRIPTION LISTENER
                    # ==========================================================================================================
                    # Re-register the handler to continue listening for new speech
                    received_handler = deepgram_live[call_sid].registerHandler(
                        deepgram_live[call_sid].event.TRANSCRIPT_RECEIVED,
                        get_transcription_add_call_sid
                    )

                else:
                    # ==========================================================================================================
                    # HANDLE PARTIAL (NON-FINAL) TRANSCRIPTIONS
                    # ==========================================================================================================
                    # This is a final transcription of a segment, but not the end of speech.
                    # Accumulate it and wait for more.
                    if gpt_talking[call_sid] == False:
                        # Accumulate partial transcription
                        if full_transcription[call_sid]:
                            full_transcription[call_sid] += " "
                        full_transcription[call_sid] += sentence_transcript
                        print(f"{call_sid} - PARTIAL TRANSCRIPTION: {sentence_transcript}")
                        print(f"{call_sid} - Accumulated transcription: {full_transcription[call_sid]}")

                        # ==========================================================================================================
                        # START TIMEOUT TIMER
                        # ==========================================================================================================
                        # Start a timer to force-send if more than 3 seconds pass without
                        # receiving speech_final. This handles cases where Deepgram
                        # doesn't detect end of speech properly.
                        if call_sid not in time_transcription:
                            time_transcription[call_sid] = time.time()

                    else:
                        # User is talking while Santa is speaking - log for potential barge-in feature
                        print(f"{call_sid} - user talking while Santa is talking, detection in progress: {sentence_transcript}")



    # ==========================================================================================================
    # REGISTER INITIAL TRANSCRIPTION HANDLER
    # ==========================================================================================================
    # Register the callback function to receive transcriptions.
    # This handler will be called every time Deepgram transcribes audio.
    received_handler = deepgram_live[call_sid].registerHandler(
        deepgram_live[call_sid].event.TRANSCRIPT_RECEIVED,
        get_transcription_add_call_sid
    )

    return deepgram_live[call_sid]


# ==========================================================================================================
# TIMER UPDATE FUNCTION
# ==========================================================================================================
def update_timer(call_sid, update_db):
    """
    Updates the remaining time for a call and optionally saves to database.

    This function calculates how much time has elapsed since the call started
    and updates the remaining time. It's called frequently during the call
    to track time usage.

    The timer is important for:
    - Billing: Limiting call duration based on user's plan
    - User experience: Allowing LLM to wrap up conversation gracefully
    - Cost control: Preventing runaway calls

    Args:
        call_sid: The unique identifier for the call
        update_db: Boolean - if True, save remaining time to database.
                   We don't always update DB to avoid excessive writes.
                   DB is updated at call end or at significant points.

    Time Display:
    - Shows remaining time every 5 seconds in HH:MM:SS format
    - Helps operators monitor call duration
    """
    # Calculate elapsed time since call started
    time_lapsed = round(time.time() - int(call_extra_info[call_sid]['start_time']))

    # Calculate remaining time (can go negative during grace period)
    call_extra_info[call_sid]['remaining_time'] = int(call_extra_info[call_sid]['timer']) - time_lapsed

    # ==========================================================================================================
    # FORMAT AND DISPLAY TIME
    # ==========================================================================================================
    # Convert remaining seconds to hours:minutes:seconds format
    hours = call_extra_info[call_sid]['remaining_time'] // 3600
    minutes = (call_extra_info[call_sid]['remaining_time'] % 3600) // 60
    seconds = call_extra_info[call_sid]['remaining_time'] % 60

    # Display remaining time every 5 seconds to avoid log spam
    if seconds % 5 == 0:
        formatted_time = f"{hours}:{minutes:02d}:{seconds:02d}"
        print(f"{call_sid} - Remaining Time: {formatted_time}")

    # ==========================================================================================================
    # DATABASE UPDATE
    # ==========================================================================================================
    if update_db:
        # Get user ID and remaining time for database update
        unique_id = call_extra_info[call_sid]['id']
        remaining_time = call_extra_info[call_sid]['remaining_time']

        try:
            conn = get_db_connection()
            cursor = conn.cursor()

            # Update the timer value in the database.
            # This persists the remaining time so it survives server restarts
            # and can be used for billing purposes.
            cursor.execute("""
                            UPDATE calls
                            SET timer = ?
                            WHERE user_id = ?
                           """, (remaining_time, unique_id))

            conn.commit()
            print(f"Timer updated correctly for ID {unique_id}.")
        except sqlite3.Error as error:
            print(f"Error updating timer in database: {error}")
        finally:
            if conn:
                conn.close()


# ==========================================================================================================
# DEEPGRAM CONNECTION CLOSE HANDLER
# ==========================================================================================================
async def handle_closing_event(ecode):
    """
    Handles the Deepgram connection close event.

    This callback is triggered when the Deepgram streaming connection closes.
    It's primarily used for error detection and debugging.

    Args:
        ecode: The close code from Deepgram
               - 1000: Normal closure (no error)
               - Other codes indicate various error conditions

    Error codes help diagnose issues like:
    - Network problems
    - Authentication failures
    - Rate limiting
    - Server errors
    """
    if (ecode != 1000):
        print(f'Connection closed with code {ecode}.')


# ==========================================================================================================
# LLM COMMUNICATION FUNCTION
# ==========================================================================================================
async def send_msg_gpt(session, call_sid, message_history, streamSid, llm_ai):
    """
    Sends the conversation history to the LLM and handles the streaming response.

    This function is the main interface to the AI models (GPT or Claude). It:
    1. Sends the complete message history to the LLM
    2. Receives the streaming response
    3. Splits the response into chunks for faster TTS processing
    4. Sends each chunk to ElevenLabs for text-to-speech conversion
    5. Updates the conversation history with the AI's response

    The streaming approach is crucial for reducing perceived latency:
    - Instead of waiting for the complete response, we start TTS conversion
      as soon as we have enough text (around 30 characters)
    - This creates a more natural, flowing conversation

    Args:
        session: The aiohttp ClientSession for making HTTP requests
        call_sid: Unique identifier for the call (for accessing correct WebSocket)
        message_history: List of all messages in the conversation so far
        streamSid: Twilio's stream identifier for sending audio back
        llm_ai: Which LLM to use ("GPT" or "Claude")

    Returns:
        tuple: (full_content, message_history)
            - full_content: The complete response from the LLM
            - message_history: Updated message history including the new response

    Response Processing:
    - Text is accumulated until it reaches ~30 characters
    - At punctuation points, text is sent to TTS
    - This balances latency vs. natural speech flow
    """
    global connected_websockets, call_extra_info

    # Get the WebSocket connection for this call
    ws = connected_websockets[call_sid]

    # ==========================================================================================================
    # INITIALIZE RESPONSE VARIABLES
    # ==========================================================================================================
    gpt_function_arguments = ""  # For function calling (not used in Santa app)
    function_name = ""           # Name of called function (if any)
    finish_reason = ""           # Why the response ended (stop, length, etc.)
    content = ""                 # Current text chunk being accumulated
    full_content = ""            # Complete response text
    previous_tts = ""            # Previous TTS chunk for context continuity

    # Token tracking for monitoring API usage and costs
    input_tokens = 0   # Tokens used in the prompt
    output_tokens = 0  # Tokens in the response
    total_tokens = 0   # Total tokens used

    # Debug identifier for tracing
    id_unico = random.uniform(1, 100000)

    # ==========================================================================================================
    # SET AI SPEAKING FLAG
    # ==========================================================================================================
    # Mark that the AI is now generating/speaking a response.
    # This prevents processing new transcriptions during response generation.
    gpt_talking[call_sid] = True

    # ==========================================================================================================
    # STREAM LLM RESPONSE
    # ==========================================================================================================
    # Iterate through each chunk of the streaming response from the LLM
    async for line in generate_chat_completion(session, call_sid, message_history):

        # ==========================================================================================================
        # OPENAI GPT RESPONSE HANDLING
        # ==========================================================================================================
        if llm_ai == "GPT":
            # OpenAI streams data with "data:" prefix
            if not line.startswith("data:"):
                continue
            else:
                # Parse the JSON data (skip "data: " prefix)
                part_data = line[6:]  # Remove "data: "

                json_data = orjson.loads(part_data)

                # Check if this is a token usage message (no response content)
                if json_data['usage'] is None:

                    finish_reason = json_data["choices"][0].get("finish_reason")

                    # "stop" means the response is complete
                    if finish_reason == "stop":
                        continue

                    # Extract the new content from this chunk
                    gpt_stream = json_data["choices"][0]["delta"]
                    new_content = gpt_stream.get("content")

                    # Accumulate content
                    full_content += new_content
                    content += filter_text(new_content)

                    # ==========================================================================================================
                    # TEXT CHUNKING FOR TTS
                    # ==========================================================================================================
                    # Call function that splits text into smaller chunks for TTS.
                    # This allows us to start speaking before the complete response is ready,
                    # reducing perceived latency.
                    content, tts_content, remaining = await insert_tts_break(content)

                    # If there's a chunk ready for TTS
                    if tts_content:
                        content = remaining

                        # ==========================================================================================================
                        # SEND CHUNK TO TEXT-TO-SPEECH
                        # ==========================================================================================================
                        # Convert this text chunk to speech using ElevenLabs.
                        # previous_text and next_text provide context for better prosody
                        # (the AI can adjust intonation knowing what came before/after).
                        await tts11AI_stream(
                            session, elevenlabs_key, tts_content, call_sid, streamSid,
                            previous_text=previous_tts, next_text=remaining
                        )
                        previous_tts = tts_content  # Update context for next chunk
                else:
                    # This is the token usage summary at the end
                    input_tokens = json_data['usage'].get('prompt_tokens')
                    output_tokens = json_data['usage'].get('completion_tokens')
                    total_tokens = json_data['usage'].get('total_tokens')
                    break

        # ==========================================================================================================
        # ANTHROPIC CLAUDE RESPONSE HANDLING
        # ==========================================================================================================
        elif llm_ai == "Claude":
            if line.startswith("data:"):
                # Parse the JSON data
                part_data = line[6:]  # Remove "data: "
                json_data = orjson.loads(part_data)
                event_type = json_data["type"]

                # Handle different Claude event types
                if event_type == "message_start":
                    # Get input token count from message start
                    input_tokens = json_data["message"]["usage"]["input_tokens"]

                elif event_type == "content_block_delta":
                    # Extract new text content
                    new_content = json_data["delta"]["text"]

                    # Accumulate content
                    full_content += new_content
                    content += filter_text(new_content)

                    # Split text for TTS (same logic as GPT)
                    content, tts_content, remaining = await insert_tts_break(content)

                    # If there's a chunk ready for TTS
                    if tts_content:
                        content = remaining

                        # Send to TTS with context
                        await tts11AI_stream(
                            session, elevenlabs_key, tts_content, call_sid, streamSid,
                            previous_text=previous_tts, next_text=remaining
                        )
                        previous_tts = tts_content

                elif event_type == "message_stop":
                    # Response complete
                    break

                elif event_type == "message_delta":
                    # Get output token count from message delta
                    output_tokens = json_data["usage"]["output_tokens"]

    # ==========================================================================================================
    # HANDLE REMAINING TEXT
    # ==========================================================================================================
    # If there's any remaining text that hasn't been sent to TTS
    # (less than 30 characters), send it now
    if (content):
        await tts11AI_stream(
            session, elevenlabs_key, content, call_sid, streamSid,
            previous_text=previous_tts
        )

    # ==========================================================================================================
    # LOG TOKEN USAGE
    # ==========================================================================================================
    total_tokens = input_tokens + output_tokens
    print(f"Tokens used {llm_ai}:\ninput_tokens: {input_tokens}\noutput_tokens: {output_tokens}\ntotal_tokens: {total_tokens}")

    # ==========================================================================================================
    # UPDATE CONVERSATION HISTORY
    # ==========================================================================================================
    # Create the assistant's message and add it to the conversation history.
    # This is crucial for maintaining context in future exchanges.
    gpt_message = {"role": "assistant", "content": full_content}
    message_history.append(gpt_message)

    # Log the complete response
    print(f"{call_sid} - {llm_ai}: {full_content}")

    # ==========================================================================================================
    # CHECK FOR HANG UP SIGNAL
    # ==========================================================================================================
    # If the phone emoji is in the response, Santa wants to hang up.
    # Send a special mark to trigger the hang up process.
    if phone_emoji in full_content:
        await send_mark_to_twilio(ws, streamSid, "hang_up")
    else:
        # Send normal TTS finished mark
        await send_mark_to_twilio(ws, streamSid)

    return full_content, message_history


# ==========================================================================================================
# TEXT CHUNKING FOR TTS
# ==========================================================================================================
async def insert_tts_break(text):
    """
    Splits text into chunks for incremental text-to-speech processing.

    This function is key to reducing perceived latency in the conversation.
    Instead of waiting for the complete LLM response before starting TTS,
    we split the text at natural break points (punctuation) and send
    chunks as soon as they're ready.

    The algorithm:
    1. If text is <= 30 characters, don't split yet (wait for more)
    2. After 30 characters, look for punctuation followed by space
    3. Split at that point and return the chunk for TTS
    4. If no punctuation found, fall back to any space

    Args:
        text: The accumulated text from the LLM response

    Returns:
        tuple: (modified_text, tts_content, remaining_content)
            - If no break needed: (original_text, None, None)
            - If break found: (text_with_marker, tts_chunk, remaining_text)

    Why 30 characters?
    - Too short: Creates choppy speech with many small segments
    - Too long: Increases latency waiting for enough text
    - 30 is a good balance for natural speech flow

    Break points:
    - Punctuation (,;!?.)]) followed by space creates natural pauses
    - These correspond to natural speech rhythms
    """
    # Don't break if we don't have enough text yet
    if len(text) <= 30:
        return (text, None, None)

    # ==========================================================================================================
    # FIND PUNCTUATION BREAK POINT
    # ==========================================================================================================
    # Search for punctuation followed by space after character 30.
    # The negative lookahead (?!\[) ensures we don't split on markers like [T_B].
    match = re.search(r"[,;!?.)\]]\s(?!\[)", text[30:])
    if match:
        # Found punctuation - split here
        punct_pos = match.start() + 30
        tts_content = text[:punct_pos+2].rstrip()  # Include punctuation and space
        remaining = text[punct_pos+2:].lstrip()     # Rest of text
        return (tts_content + '[T_B]' + remaining, tts_content, remaining)

    # ==========================================================================================================
    # FALLBACK: FIND ANY SPACE
    # ==========================================================================================================
    # If no punctuation found, split at any space to avoid cutting words
    match = re.search(r"\s+", text[30:])
    if match:
        space_pos = match.start() + 30
        tts_content = text[:space_pos].rstrip()
        remaining = text[space_pos:].lstrip()
        return (tts_content + '[T_B]' + remaining, tts_content, remaining)

    # No good break point found - keep accumulating
    return (text, None, None)


# ==========================================================================================================
# MP3 TO TWILIO AUDIO CONVERSION
# ==========================================================================================================
async def send_mp3_to_twilio(mp3_data: bytes, call_sid: str, stream_sid: str):
    """
    Converts MP3 audio to Twilio's format and sends it to the active call.

    This function handles pre-recorded audio files (like Santa's intro).
    Twilio requires a very specific audio format, so we need to convert:
    - From: MP3 (common audio format)
    - To: Mulaw (μ-law) 8-bit encoding at 8000Hz mono

    The conversion process:
    1. Load MP3 into pydub AudioSegment
    2. Convert to WAV format
    3. Resample to 8000Hz, mono, 16-bit
    4. Convert to μ-law encoding using audioop
    5. Stream in 512-byte chunks to Twilio

    Args:
        mp3_data: Raw bytes of the MP3 file
        call_sid: Identifier for the call (to find WebSocket)
        stream_sid: Twilio's stream identifier for routing audio

    Why this format?
    - Twilio's telephony system uses traditional phone audio standards
    - μ-law is the standard encoding for North American phone systems
    - 8000Hz is the standard sample rate for telephone audio

    Chunk size:
    - 512 bytes is optimal for real-time streaming
    - Balances latency vs. overhead
    """
    global connected_websockets

    # Get the WebSocket connection for this call
    ws = connected_websockets[call_sid]

    # ==========================================================================================================
    # CONVERT MP3 TO WAV
    # ==========================================================================================================
    # Load MP3 from byte data using pydub
    audio = AudioSegment.from_file(io.BytesIO(mp3_data), format="mp3")

    # Export to WAV format (intermediate format for processing)
    audio_wav = io.BytesIO()
    audio.export(audio_wav, format="wav")
    audio_wav.seek(0)

    # ==========================================================================================================
    # CONVERT TO TWILIO FORMAT
    # ==========================================================================================================
    # Resample audio to Twilio's requirements:
    # - 8000Hz sample rate (telephone quality)
    # - 16-bit samples (for μ-law conversion)
    # - Mono (single channel)
    audio = AudioSegment.from_wav(audio_wav)
    audio = audio.set_frame_rate(8000).set_sample_width(2).set_channels(1)

    # ==========================================================================================================
    # CONVERT TO MULAW ENCODING
    # ==========================================================================================================
    # μ-law (mu-law) is a companding algorithm used in telephony.
    # It provides better dynamic range in 8 bits than linear encoding.
    audio_data = audio.raw_data
    audio_data_mulaw = audioop.lin2ulaw(audio_data, audio.sample_width)

    # ==========================================================================================================
    # STREAM TO TWILIO
    # ==========================================================================================================
    # Send audio in 512-byte chunks for smooth streaming
    CHUNK_SIZE = 512
    for i in range(0, len(audio_data_mulaw), CHUNK_SIZE):
        chunk = audio_data_mulaw[i:i+CHUNK_SIZE]
        await send_audio_to_twilio(ws, chunk, stream_sid)


# ==========================================================================================================
# ROLE PROMPT READER
# ==========================================================================================================
def read_role_prompt(file_path, call_sid):
    """
    Reads and personalizes the role prompt for the AI.

    The role prompt defines Santa's personality, behavior, and knowledge.
    This function loads the template and substitutes placeholders with
    actual values specific to this call.

    Personalization includes:
    - Child's name: So Santa can address them personally
    - Parents' names: So Santa can mention them
    - Gifts: What the child asked for (for Santa to reference)
    - Context: Special situations or notes about the child
    - Language: For proper response language
    - Date/time: For contextual awareness
    - Secret words: For special control commands

    Args:
        file_path: Path to the role prompt template file
        call_sid: Call identifier (for accessing call-specific data)

    Returns:
        str: The personalized prompt string

    Template Placeholders:
    - {call_sid}: Unique call identifier
    - {current_datetime}: Current date/time
    - {child_name}: Name of the child
    - {father_name}: Name of the father
    - {mother_name}: Name of the mother
    - {regalos}: Gifts the child wants
    - {contexto}: Special context/notes
    - {num_language}: Language code (en, es)
    - {secret_instruction_word}: Admin control word
    - {secret_exit_word}: Emergency exit word
    """
    global call_extra_info

    # ==========================================================================================================
    # GET SECRET WORDS FROM ENVIRONMENT
    # ==========================================================================================================
    # Secret words allow administrators to control the AI during calls.
    # These should be unique and unpredictable for security.
    secret_instruction_word = os.getenv("SECRET_INSTRUCTION_WORD", "DefaultSecretWord1")
    secret_exit_word = os.getenv("SECRET_EXIT_WORD", "DefaultSecretWord2")

    # ==========================================================================================================
    # READ AND PERSONALIZE TEMPLATE
    # ==========================================================================================================
    with open(file_path, "r", encoding='utf-8') as file:
        content = file.read().strip()

        # Substitute all placeholders with actual values
        return content.format(
            call_sid=call_sid,
            current_datetime=call_extra_info[call_sid]['current_datetime'],
            child_name=call_extra_info[call_sid]['child_name'],
            father_name=call_extra_info[call_sid]['father_name'],
            mother_name=call_extra_info[call_sid]['mother_name'],
            regalos=call_extra_info[call_sid]['gifts'],
            contexto=call_extra_info[call_sid]['context'],
            num_language=call_extra_info[call_sid]['lang'],
            secret_instruction_word=secret_instruction_word,
            secret_exit_word=secret_exit_word
        )


# ==========================================================================================================
# CONVERSATION INITIALIZATION
# ==========================================================================================================
def initialize_role_message(call_sid):
    """
    Initializes the conversation history for a new call.

    This function sets up the initial state of the conversation including:
    - The system prompt (role definition for the AI)
    - Initial "fake" conversation context

    The fake conversation gives the AI context that the call has just
    been answered with Santa's greeting. This allows natural continuation
    when the child responds.

    Args:
        call_sid: Unique identifier for this call

    Conversation Structure:
    - For GPT: System message goes in message history
    - For Claude: System message is sent separately (in call_extra_info)

    Initial Messages:
    - User: "Hello?" (simulating child answering)
    - Assistant: "Ho, Ho, Ho! I'm Santa..." (Santa's greeting)

    This setup means when the real conversation starts, the AI knows:
    - It's playing Santa Claus
    - The call has just been answered
    - It's time to continue the conversation naturally
    """
    global conversations, from_call_sid, query_params

    # Only initialize if not already exists (prevents resetting mid-call)
    if call_sid not in conversations:
        # Read and personalize the role prompt
        role_prompt = read_role_prompt(role_file_path, call_sid)

        # ==========================================================================================================
        # SET UP INITIAL CONVERSATION CONTEXT
        # ==========================================================================================================
        # These initial messages provide context to the AI.
        # They simulate the call being answered and Santa's initial greeting.
        # The next message the AI receives will be the actual child's response.
        conversations[call_sid] = [
            {"role": "user", "content": "Hello?"},
            {"role": "assistant", "content": "Ho, Ho, Ho! I'm Santa Claus, who do I have the pleasure of speaking with?"}
        ]

        # ==========================================================================================================
        # ADD SYSTEM PROMPT (LLM-SPECIFIC)
        # ==========================================================================================================
        # GPT requires the system prompt as the first message in history
        if llm_ai == "GPT":
            conversations[call_sid].insert(0, {"role": "system", "content": role_prompt})
        # Claude requires the system prompt to be sent separately
        elif llm_ai == "Claude":
            call_extra_info[call_sid]["prompt"] = role_prompt



# ==========================================================================================================
# LLM API CALL GENERATOR
# ==========================================================================================================
async def generate_chat_completion(session, call_sid, message_history, model=model_ai, temperature=0.5, max_tokens=1024, streaming=True):
    """
    Makes the API call to the LLM and yields streaming response chunks.

    This is an async generator function that:
    1. Prepares the API request based on the LLM provider (GPT or Claude)
    2. Makes the HTTP request to the API
    3. Yields each chunk of the streaming response

    Using streaming responses is crucial for low-latency conversations:
    - We start processing text as soon as it arrives
    - TTS can begin before the complete response is ready
    - This significantly reduces perceived response time

    Args:
        session: aiohttp ClientSession for HTTP requests
        call_sid: Call identifier (for accessing call-specific data)
        message_history: Complete conversation history
        model: Which AI model to use (default from config)
        temperature: Response randomness (0.0-1.0, higher = more creative)
        max_tokens: Maximum response length
        streaming: Whether to stream the response (always True for calls)

    Yields:
        str: Each line/chunk of the streaming response

    API Differences:
    - GPT: Uses Authorization header with Bearer token
    - Claude: Uses x-api-key header with anthropic-version
    - GPT: System prompt in messages array
    - Claude: System prompt as separate "system" field
    """
    # ==========================================================================================================
    # PREPARE GPT REQUEST
    # ==========================================================================================================
    if llm_ai == "GPT":
        url = f"https://api.openai.com/v1/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {openai_key}",
        }
        data = {
            "model": model,
            "messages": message_history,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": True,
            "stream_options": {"include_usage": True}  # Get token counts at end
        }

    # ==========================================================================================================
    # PREPARE CLAUDE REQUEST
    # ==========================================================================================================
    elif llm_ai == "Claude":
        url = "https://api.anthropic.com/v1/messages"
        headers = {
            "Content-Type": "application/json",
            "x-api-key": claude_key,
            "anthropic-version": "2023-06-01",
            "anthropic-beta": "messages-2023-12-15"
        }
        data = {
            "model": model,
            "system": call_extra_info[call_sid]["prompt"],  # Claude takes system separately
            "max_tokens": max_tokens,
            "messages": message_history,
            "temperature": temperature,
            "stream": True
        }

    # ==========================================================================================================
    # MAKE STREAMING REQUEST
    # ==========================================================================================================
    # Use the session's connection pool to make the request.
    # ssl=False is used here (should be True in production for security).
    # Each chunk of the response is decoded and yielded immediately.
    async with session[call_sid].post(url, ssl=False, headers=headers, data=orjson.dumps(data)) as response:
        async for chunk in response.content:
            yield chunk.decode("utf-8")



# ==========================================================================================================
# AUDIO TRANSMISSION TO TWILIO
# ==========================================================================================================
async def send_audio_to_twilio(ws: WebSocket, chunk: bytes, streamSid: str):
    """
    Sends an audio chunk to Twilio via WebSocket.

    This function packages audio data in Twilio's expected format and
    sends it through the WebSocket connection. The audio will be played
    to the caller in real-time.

    Twilio's streaming protocol requires:
    - event: "media" for audio data
    - streamSid: To identify which stream to play on
    - media.payload: Base64-encoded audio data

    Args:
        ws: The WebSocket connection to Twilio
        chunk: Raw audio bytes (already in mulaw format)
        streamSid: Twilio's stream identifier

    Data Format:
    - Audio must be in mulaw, 8000Hz, mono format
    - Payload must be Base64 encoded
    - JSON structure must match Twilio's specification
    """
    media_data = {
        "event": "media",
        "streamSid": streamSid,
        "media": {
            "payload": base64.b64encode(chunk).decode("utf-8")
        }
    }
    await ws.send_json(media_data)


# ==========================================================================================================
# MARK EVENT TO TWILIO
# ==========================================================================================================
async def send_mark_to_twilio(ws: WebSocket, streamSid: str, name = "TTS_Finished"):
    """
    Sends a mark event to Twilio to signal audio playback completion.

    Mark events are used to synchronize with Twilio. When Twilio finishes
    playing all queued audio, it sends back the mark event, allowing us
    to know when it's safe to accept new input.

    Use cases:
    - "TTS_Finished": Normal end of AI speech, ready for user input
    - "hang_up": Signal that the call should be terminated

    Args:
        ws: The WebSocket connection to Twilio
        streamSid: Twilio's stream identifier
        name: The mark name (default: "TTS_Finished")

    This synchronization is crucial for natural conversation flow:
    - We don't want to process new transcriptions while audio is playing
    - The mark event tells us when Santa has finished speaking
    """
    media_data = {
        "event": "mark",
        "streamSid": streamSid,
        "mark": {
            "name": name
        }
    }
    await ws.send_json(media_data)


# ==========================================================================================================
# TEXT FILTERING FOR TTS
# ==========================================================================================================
def filter_text(text):
    """
    Filters text to only include characters safe for TTS.

    This function removes any characters that could cause issues with
    text-to-speech processing. This includes:
    - Emojis (except specific control ones)
    - Special Unicode characters
    - Potential code injection attempts

    The filter allows:
    - ASCII letters (a-z, A-Z)
    - Digits (0-9)
    - Whitespace (spaces, newlines, tabs)
    - Standard punctuation
    - Spanish accented characters (áéíóúÁÉÍÓÚüÜñÑ)

    Args:
        text: The raw text from the LLM

    Returns:
        str: Cleaned text safe for TTS processing

    Security Note:
    - This helps prevent prompt injection through TTS
    - Removes potential control characters
    - Ensures predictable audio output
    """
    # Define allowed characters
    allowed_characters = string.ascii_letters + string.digits + string.whitespace + string.punctuation + "áéíóúÁÉÍÓÚüÜñÑ"

    # Create regex pattern for disallowed characters
    pattern = f"[^{re.escape(allowed_characters)}]"

    # Remove all disallowed characters
    filtered_text = re.sub(pattern, "", text)
    return filtered_text


# ==========================================================================================================
# ELEVENLABS TEXT-TO-SPEECH STREAMING
# ==========================================================================================================
async def tts11AI_stream(session, key: str, text: str, call_sid, streamSid, voice_id: str = None, stability: float = 0.59, similarity_boost: float = 0.99, previous_text: str = "", next_text: str = ""):
    """
    Converts text to speech using ElevenLabs API and streams to Twilio.

    ElevenLabs provides high-quality, natural-sounding voice synthesis.
    This function:
    1. Sends text to ElevenLabs API with voice settings
    2. Receives audio in streaming chunks
    3. Converts audio to Twilio's mulaw format
    4. Sends audio to the active call

    Voice Parameters:
    - voice_id: The specific voice to use (configurable)
    - stability: Voice consistency (0.0-1.0)
      - Higher: More consistent but potentially monotone
      - Lower: More expressive but may vary
    - similarity_boost: How closely to match the original voice

    Context Parameters (for better prosody):
    - previous_text: What was just said (helps with intonation)
    - next_text: What comes next (helps with continuation)

    Args:
        session: aiohttp session for HTTP requests
        key: ElevenLabs API key
        text: Text to convert to speech
        call_sid: Call identifier
        streamSid: Twilio stream identifier
        voice_id: ElevenLabs voice ID (optional, uses env default)
        stability: Voice stability parameter
        similarity_boost: Voice similarity parameter
        previous_text: Previous text for context
        next_text: Upcoming text for context

    Audio Flow:
    1. Text → ElevenLabs API → μ-law audio stream
    2. μ-law audio → Twilio WebSocket → Caller's phone
    """
    global connected_websockets, call_extra_info

    # Debug identifier
    id_unico = random.uniform(1, 100000)

    # ==========================================================================================================
    # GET VOICE ID FROM ENVIRONMENT IF NOT PROVIDED
    # ==========================================================================================================
    if voice_id is None:
        voice_id = os.getenv("ELEVENLABS_VOICE_ID", "Gqe8GJJLg3haJkTwYj2L")

    # ==========================================================================================================
    # VERIFY CONNECTION STATE
    # ==========================================================================================================
    # Check that the WebSocket and stream are still active.
    # The call might have ended while we were generating the response.
    ws = connected_websockets.get(call_sid)
    if ws is None:
        return

    if streamSid is None:
        return

    # ==========================================================================================================
    # PREPARE ELEVENLABS API REQUEST
    # ==========================================================================================================
    # Request μ-law format directly to avoid additional conversion.
    # optimize_streaming_latency=3 enables maximum latency optimization.
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}/stream?optimize_streaming_latency=3&output_format=ulaw_8000"

    headers = {
        "Accept": "audio/basic",        # Request raw audio
        "Content-Type": "application/json",
        "xi-api-key": key
    }

    # ==========================================================================================================
    # SELECT TTS MODEL
    # ==========================================================================================================
    # eleven_turbo_v2_5 provides the best balance of quality and speed
    # for real-time applications. Other options:
    # - eleven_turbo_v2: Fast, English-optimized
    # - eleven_multilingual_v2: Better for non-English
    model_id = "eleven_turbo_v2_5"

    # ==========================================================================================================
    # PREPARE REQUEST DATA
    # ==========================================================================================================
    data = {
        "text": text,
        "model_id": model_id,
        "voice_settings": {
            "stability": stability,
            "similarity_boost": similarity_boost
        }
    }

    # ==========================================================================================================
    # ADD CONTEXT FOR BETTER PROSODY
    # ==========================================================================================================
    # ElevenLabs can use surrounding text context to improve prosody.
    # This makes speech sound more natural when split across chunks.
    if previous_text:
        data["previous_text"] = previous_text
    if next_text:
        data["next_text"] = next_text

    CHUNK_SIZE = 1024  # Audio chunk size for streaming

    # ==========================================================================================================
    # STREAM AUDIO TO TWILIO
    # ==========================================================================================================
    # Make request to ElevenLabs and stream audio directly to Twilio.
    # This minimizes memory usage and latency.
    async with session[call_sid].post(url, ssl=False, data=orjson.dumps(data), headers=headers) as response:
        async for chunk in response.content.iter_chunked(CHUNK_SIZE):
            if chunk:
                await send_audio_to_twilio(ws, chunk, streamSid)


# ==========================================================================================================
# CALL HANG UP FUNCTION
# ==========================================================================================================
async def hang_up_call(call_sid):
    """
    Terminates an active phone call.

    This function uses Twilio's API to hang up a call by updating it
    with TwiML (Twilio's XML-based markup language) that instructs
    Twilio to terminate the call.

    Args:
        call_sid: Twilio's unique identifier for the call to terminate

    The function handles TwilioRestException silently because:
    - The call might have already ended
    - Network issues might prevent the update
    - In either case, we don't want to crash the application

    TwiML Used:
    <Response><Hangup/></Response>
    - This is the simplest TwiML to terminate a call
    """
    print(f"Hanging up call")
    try:
        call = twilio_client.calls(call_sid).update(
            twiml=f'<Response><Hangup/></Response>'
        )

    except TwilioRestException as e:
        # Silently handle errors (call may already be ended)
        pass


# ==========================================================================================================
# WEBSOCKET STREAM HANDLER
# ==========================================================================================================
@app.websocket("/stream/{call_sid}")
async def stream(websocket: WebSocket, call_sid: str):
    """
    Main WebSocket handler for bidirectional audio streaming with Twilio.

    This is the core function that handles the real-time audio conversation.
    When a call is answered, Twilio connects to this WebSocket endpoint
    and streams audio bidirectionally:
    - Inbound: Caller's voice → This server (for transcription)
    - Outbound: This server → Caller (Santa's voice)

    The function:
    1. Accepts the WebSocket connection
    2. Handles the "start" event to initialize the call
    3. Receives audio in "media" events and sends to Deepgram
    4. Handles "stop" and "mark" events for call flow control
    5. Cleans up resources when the call ends

    Args:
        websocket: The FastAPI WebSocket connection
        call_sid: The Twilio call identifier (from URL path)

    Event Types:
    - start: Call connection established, contains metadata
    - media: Audio data (base64 encoded)
    - stop: Stream is ending
    - mark: Marker event (used for synchronization)

    State Management:
    - Initializes conversation history
    - Sets up Deepgram transcription
    - Manages speaking/listening states
    """
    global deepgram_live
    global connected_websockets, conversations, call_extra_info, connector, session

    local_websockets = websocket

    # ==========================================================================================================
    # ACCEPT WEBSOCKET CONNECTION
    # ==========================================================================================================
    # Accept the incoming WebSocket connection from Twilio
    await websocket.accept()

    # Initialize speaking flag (Santa not speaking at start)
    gpt_talking[call_sid] = False
    continue_loop = True

    try:
        # ==========================================================================================================
        # MAIN CONNECTION LOOP
        # ==========================================================================================================
        # Outer loop keeps connection alive
        while continue_loop:

            # ==========================================================================================================
            # MESSAGE PROCESSING LOOP
            # ==========================================================================================================
            # Inner loop processes each message
            while True:
                # Receive text message from Twilio
                message = await websocket.receive_text()

                # Skip empty messages
                if message is None:
                    continue

                # Parse JSON message
                data = json.loads(message)

                # ==========================================================================================================
                # HANDLE "START" EVENT
                # ==========================================================================================================
                # The start event is sent once when the stream is established.
                # It contains important metadata about the call.
                if data['event'] == "start":

                    print(f"Connected START received, call_sid: {call_sid}")

                    # Extract parameters from the start event
                    parameters = data['start']
                    custom_parameters = data['start'].get('customParameters')

                    # Process custom parameters (phone numbers for logging)
                    if custom_parameters:
                        num_from = custom_parameters.get("num_from")
                        print(f"num_from: {num_from}")

                        num_to = custom_parameters.get("num_to")
                        print(f"num_to: {num_to}")

                    # ==========================================================================================================
                    # RECORD CALL START TIME
                    # ==========================================================================================================
                    # Store the start time for timer calculations
                    call_extra_info[call_sid].update({"start_time": time.time()})

                    # Store WebSocket reference for this call
                    connected_websockets[call_sid] = websocket

                    # Get the stream SID from Twilio
                    stream_sid = data['start']['streamSid']

                    # ==========================================================================================================
                    # INITIALIZE DEEPGRAM
                    # ==========================================================================================================
                    # Set up speech-to-text transcription for this call
                    deepgram_live[call_sid] = await setup_deepgram_sdk(call_sid, stream_sid)

                    # Check if Deepgram initialized successfully
                    if deepgram_live[call_sid]:
                        continue_loop = True
                    else:
                        # Deepgram failed - hang up the call
                        continue_loop = False
                        print("Error: Could not start deepgram_live")
                        await hang_up_call(call_sid)
                        break

                    # ==========================================================================================================
                    # INITIALIZE CONVERSATION
                    # ==========================================================================================================
                    # Only initialize if this is a new call (not reconnecting)
                    if call_sid not in conversations:

                        # Set up the conversation history with Santa's persona
                        initialize_role_message(call_sid)
                        message_history = conversations[call_sid]

                        # ==========================================================================================================
                        # PLAY INTRO AUDIO
                        # ==========================================================================================================
                        # Play pre-recorded intro while AI prepares.
                        # This reduces perceived latency at call start.
                        gpt_talking[call_sid] = True
                        if call_extra_info[call_sid]['lang'] == "es":
                            await send_mp3_to_twilio(intro_spanish_mp3, call_sid, stream_sid)
                        else:
                            await send_mp3_to_twilio(intro_english_mp3, call_sid, stream_sid)

                        # Send mark to know when intro finishes
                        await send_mark_to_twilio(websocket, stream_sid)

                        # ==========================================================================================================
                        # CREATE HTTP SESSION POOL
                        # ==========================================================================================================
                        # Create connection pool for API requests (LLM, TTS)
                        connector[call_sid], session[call_sid] = await create_pool()

                # ==========================================================================================================
                # HANDLE "MEDIA" EVENT
                # ==========================================================================================================
                # Media events contain audio data from the caller
                if data['event'] == "media":
                    # Decode base64 audio
                    audio = base64.b64decode(data['media']['payload'])

                    # Only process inbound (caller → server) audio
                    if data['media']['track'] == 'inbound':
                        # Send to Deepgram for transcription
                        if deepgram_live[call_sid]:
                            deepgram_live[call_sid].send(audio)

                # ==========================================================================================================
                # HANDLE "STOP" EVENT
                # ==========================================================================================================
                # Stop event signals the stream is ending
                if data['event'] == "stop":
                    break

                # ==========================================================================================================
                # HANDLE "MARK" EVENT
                # ==========================================================================================================
                # Mark events are used for synchronization
                if data['event'] == "mark":
                    # Check which mark was received
                    if data['mark']['name'] == "hang_up":
                        # Hang up signal received - terminate call
                        await hang_up_call(call_sid)
                        continue
                    elif data['mark']['name'] == "TTS_Finished":
                        # TTS finished playing - ready for user input
                        gpt_talking[call_sid] = False
                        continue

                # End of message processing loop

    # ==========================================================================================================
    # HANDLE WEBSOCKET DISCONNECT
    # ==========================================================================================================
    except WebSocketDisconnect:
        # Normal disconnect - the call has ended
        pass

    # ==========================================================================================================
    # HANDLE UNEXPECTED ERRORS
    # ==========================================================================================================
    except Exception as e:
        print(f"error: {str(e)}")

    # ==========================================================================================================
    # CLEANUP RESOURCES
    # ==========================================================================================================
    finally:
        print("Exiting")

        # Update timer in database if call was successful
        if continue_loop:
            update_timer(call_sid, True)

        # Close Deepgram connection
        if deepgram_live[call_sid]:
            await deepgram_live[call_sid].finish()

        # Clean up all call-specific data structures
        if connected_websockets.get(call_sid):
            if call_sid in full_transcription:
                del full_transcription[call_sid]

            if call_sid in call_extra_info:
                del call_extra_info[call_sid]

            if call_sid in connected_websockets:
                del connected_websockets[call_sid]

        # Close HTTP session and connector
        if call_sid in session:
            await session[call_sid].close()
            del session[call_sid]

        if call_sid in connector:
            await connector[call_sid].close()
            del connector[call_sid]


# ==========================================================================================================
# INCOMING CALL HANDLER
# ==========================================================================================================
@app.post('/answer')
async def handle_incoming_call(request: Request):
    """
    Handles incoming calls to the Twilio number.

    This endpoint is triggered when someone calls the Twilio phone number.
    It responds with TwiML instructions to play an intro audio.

    Note: This is for incoming calls (someone calling Santa), not outgoing
    calls (Santa calling children). The main use case is outgoing calls
    handled by /answer2 and /answer.

    Args:
        request: FastAPI Request object

    Returns:
        TwiML response instructing Twilio to play the intro audio
    """
    response = VoiceResponse()
    response.play(intro_audio_url)

    return Response(content=str(response), media_type='text/xml')


# ==========================================================================================================
# OUTGOING CALL HANDLER (JSON FILE METHOD)
# ==========================================================================================================
@app.post('/answer2/{user_json}')
async def answer2(request: Request, user_json: str):
    """
    Handles outgoing calls initiated via caller.py with JSON file data.

    This endpoint is called by Twilio when a call initiated by caller.py
    is answered. It reads user data from a JSON file and sets up the
    call stream.

    Args:
        request: FastAPI Request object containing Twilio form data
        user_json: Filename of the JSON file containing user data

    Returns:
        TwiML response with Stream connection instructions

    Flow:
    1. Receive Twilio form data (call info)
    2. Load user data from JSON file
    3. Store user data in call_extra_info
    4. Return TwiML to connect to WebSocket stream

    The Stream connection redirects audio to our WebSocket handler
    where the actual conversation takes place.
    """
    global from_call_sid, query_params, call_extra_info, websockets_url

    # ==========================================================================================================
    # GET TWILIO FORM DATA
    # ==========================================================================================================
    form_data = await request.form()
    query_params = request.query_params

    # Debug: print URL parameters
    for param in query_params.keys():
        print(f'URL Param - {param}: {query_params[param]}')

    # Get call SID from Twilio
    call_sid = form_data["CallSid"]

    # ==========================================================================================================
    # LOAD USER DATA FROM JSON FILE
    # ==========================================================================================================
    with open(f"users/{user_json}", 'r') as json_file:
        user_data = json.load(json_file)

    # Store user data in call_extra_info dictionary
    call_extra_info[call_sid] = user_data
    call_extra_info[call_sid].update({"user_json": user_json})

    # Get phone numbers for logging
    num_from = form_data["From"]
    num_to = form_data["To"]

    # ==========================================================================================================
    # BUILD TWIML RESPONSE
    # ==========================================================================================================
    response = VoiceResponse()
    connect = Connect()

    # Create stream connection to our WebSocket handler
    stream = Stream(url=f'wss://{websockets_url}/stream/{call_sid}')

    # Add custom parameters to be sent with the stream
    stream.parameter(name='num_from', value=f'{num_from}')
    stream.parameter(name='num_to', value=f'{num_to}')

    connect.append(stream)
    response.append(connect)

    return Response(content=str(response), media_type='text/xml')


# ==========================================================================================================
# OUTGOING CALL HANDLER (DATABASE METHOD)
# ==========================================================================================================
@app.post('/answer/{user_id}/{call_job_id}')
async def answer(request: Request, user_id: str, call_job_id: str):
    """
    Handles outgoing calls initiated via the scheduler with database data.

    This endpoint is called by Twilio when a scheduled call is answered.
    It retrieves user data from the database and sets up the call stream.

    Args:
        request: FastAPI Request object containing Twilio form data
        user_id: User ID from the database
        call_job_id: Scheduled job ID (for verification)

    Returns:
        TwiML response with Stream connection instructions

    Flow:
    1. Receive Twilio form data
    2. Verify user_id and call_job_id match in database
    3. Load user details from database
    4. Calculate timezone and current datetime
    5. Store data in call_extra_info
    6. Return TwiML to connect to WebSocket stream

    Security:
    - Validates that user_id and call_job_id match
    - Prevents unauthorized access to other users' calls
    """
    global call_extra_info, websockets_url

    # Get form data from Twilio
    form_data = await request.form()
    call_sid = form_data["CallSid"]

    # ==========================================================================================================
    # DATABASE VERIFICATION AND DATA RETRIEVAL
    # ==========================================================================================================
    conn = get_db_connection()
    c = conn.cursor()

    # Query to verify user_id and call_job_id match, and get all user data
    c.execute("""
                SELECT user_details.*, calls.call_job_id, calls.timer, users.lang, calls.time_zone
                FROM users
                INNER JOIN user_details ON users.id = user_details.user_id
                INNER JOIN calls ON users.id = calls.user_id
                WHERE users.id = ? AND calls.call_job_id = ?
              """, (user_id, call_job_id))

    user_data = c.fetchone()

    # Return 404 if user or job not found
    if not user_data:
        conn.close()
        print("user or job not found")
        raise HTTPException(status_code=404, detail="User or job not found")

    # Convert to dictionary for easier access
    user_dict = dict(zip([column[0] for column in c.description], user_data))
    conn.close()

    # ==========================================================================================================
    # TIMEZONE HANDLING
    # ==========================================================================================================
    # Get user's timezone and calculate current datetime
    user_timezone = pytz.timezone(user_dict['time_zone'])
    current_datetime = datetime.now(user_timezone)
    current_datetime_str = current_datetime.strftime('%Y-%m-%d')

    # Add datetime info to user dictionary
    user_dict['current_datetime'] = current_datetime_str
    user_dict['time_zone'] = user_dict['time_zone']

    # Store user data for this call
    call_extra_info[call_sid] = user_dict

    num_from = form_data["From"]
    num_to = form_data["To"]

    # ==========================================================================================================
    # BUILD TWIML RESPONSE
    # ==========================================================================================================
    response = VoiceResponse()
    connect = Connect()
    stream = Stream(url=f'wss://{websockets_url}/stream/{call_sid}')

    # Add parameters to stream
    stream.parameter(name='user_id', value=user_id)
    stream.parameter(name='num_from', value=num_from)
    stream.parameter(name='num_to', value=num_to)

    connect.append(stream)
    response.append(connect)

    return Response(content=str(response), media_type='text/xml')


# ==========================================================================================================
# CALL SCHEDULING ENDPOINT
# ==========================================================================================================
@app.post("/schedule-call")
async def schedule_call(request: Request, user_id: str = None, call_date: str = None, call_time: str = None, time_zone: str = None, conn = None, close_conn = True):
    """
    Schedules a call for a specific user at a specific time.

    This endpoint creates a scheduled job that will initiate a call
    at the specified date and time in the user's timezone.

    Args:
        request: FastAPI Request object
        user_id: User ID to schedule call for (can come from body or param)
        call_date: Date for the call (YYYY-MM-DD) - optional, uses DB if not provided
        call_time: Time for the call (HH:MM) - optional, uses DB if not provided
        time_zone: User's timezone - optional, uses DB if not provided
        conn: Database connection (optional, for batching)
        close_conn: Whether to close connection (for batching operations)

    Returns:
        JSONResponse with job_id and success message

    Security:
    - Only accessible from localhost (internal use only)
    - Prevents external parties from scheduling calls

    Flow:
    1. Verify request is from localhost
    2. Get user_id from request body or parameter
    3. Load call details from database
    4. Convert user's timezone to server timezone
    5. Schedule job with APScheduler
    6. Store job_id in database for tracking
    """
    # ==========================================================================================================
    # SECURITY CHECK
    # ==========================================================================================================
    # Only allow internal requests from localhost
    if not is_localhost(request):
        raise HTTPException(status_code=403, detail="Access denied: internal endpoint only")

    # ==========================================================================================================
    # GET USER ID
    # ==========================================================================================================
    if user_id is None:
        body = await request.json()
        user_id_from_post = body.get('user_id', None)
        user_id = user_id_from_post if user_id_from_post else user_id

        if user_id is None:
            raise HTTPException(status_code=400, detail="user_id is required")

    # ==========================================================================================================
    # LOAD USER DATA FROM DATABASE
    # ==========================================================================================================
    if conn is None:
        conn = get_db_connection()
    try:
        c = conn.cursor()
        c.execute("""
            SELECT users.*, user_details.phone_number, user_details.child_name, user_details.father_name, user_details.mother_name, user_details.gifts, user_details.context, calls.call_date, calls.call_time, calls.time_zone, calls.timer
            FROM users
            JOIN user_details ON users.id = user_details.user_id
            JOIN calls ON users.id = calls.user_id
            WHERE users.id=?
        """, (user_id,))
        row = c.fetchone()
        if row:
            user_data = dict(zip([column[0] for column in c.description], row))
            call_date = user_data['call_date']
            call_time = user_data['call_time']
            time_zone = user_data['time_zone']
        else:
            print("user_id not found")
            return JSONResponse(content={'message': 'User not found'}, status_code=404)
    except sqlite3.Error as e:
        print(f"Read Database error: {e}")
        return JSONResponse(content={'message': ['error500']}, status_code=500)

    # ==========================================================================================================
    # TIMEZONE CONVERSION
    # ==========================================================================================================
    # Convert user's local time to server timezone (Europe/Madrid)
    call_datetime_str = f"{call_date} {call_time}"
    call_datetime = datetime.strptime(call_datetime_str, '%Y-%m-%d %H:%M')

    user_timezone = pytz.timezone(time_zone)
    user_datetime = user_timezone.localize(call_datetime)

    # Convert to server timezone
    converted_timezone = pytz.timezone('Europe/Madrid')
    converted_datetime = user_datetime.astimezone(converted_timezone)

    # ==========================================================================================================
    # SCHEDULE THE JOB
    # ==========================================================================================================
    job_id = str(uuid.uuid4())
    print(f"Scheduling call for User ID: {user_id} at {converted_datetime.strftime('%Y-%m-%d %H:%M:%S %Z')} with Job ID: {job_id}")

    # Add job to scheduler
    job = scheduler.add_job(
        func=initiate_call,
        trigger='date',
        run_date=converted_datetime,
        args=[user_id],
        id=job_id
    )

    # ==========================================================================================================
    # STORE JOB ID IN DATABASE
    # ==========================================================================================================
    if conn is None:
        conn = get_db_connection()

    try:
        c = conn.cursor()
        c.execute("UPDATE calls SET call_job_id=? WHERE user_id=?", (job.id, user_id))
        conn.commit()
    except sqlite3.Error as e:
        print(f"Update Database error: {e}")
        return JSONResponse(content={'message': strings_data['error500']}, status_code=500)
    finally:
        if close_conn:
            conn.close()

    return JSONResponse(content={'message': 'Call scheduled successfully', 'job_id': job_id}, status_code=200)


# ==========================================================================================================
# CALL CANCELLATION ENDPOINT
# ==========================================================================================================
@app.post("/cancel-call")
async def cancel_call(request: Request, request_body: CancelCallRequest):
    """
    Cancels a scheduled call for a user.

    This endpoint removes a scheduled call from the APScheduler and
    clears the scheduling information from the database.

    Args:
        request: FastAPI Request object
        request_body: Pydantic model containing user_id

    Returns:
        JSONResponse indicating success or failure

    Security:
    - Only accessible from localhost (internal use only)

    Validation:
    - Checks if call has already happened
    - Verifies user exists in database
    """
    # ==========================================================================================================
    # SECURITY CHECK
    # ==========================================================================================================
    if not is_localhost(request):
        raise HTTPException(status_code=403, detail="Access denied: internal endpoint only")

    user_id = request_body.user_id
    conn = get_db_connection()
    conn.row_factory = sqlite3.Row

    try:
        c = conn.cursor()
        c.execute("SELECT call_job_id, call_date, call_time, time_zone FROM calls WHERE user_id=?", (user_id,))
        row = c.fetchone()

        if row:
            job_id, call_date, call_time, time_zone = row

            # ==========================================================================================================
            # CHECK IF CALL ALREADY HAPPENED
            # ==========================================================================================================
            call_datetime_str = f"{call_date} {call_time}"
            call_datetime = datetime.strptime(call_datetime_str, '%Y-%m-%d %H:%M')

            user_timezone = pytz.timezone(time_zone)
            user_datetime = user_timezone.localize(call_datetime)

            current_datetime = datetime.now(pytz.timezone(time_zone))
            if current_datetime > user_datetime:
                return JSONResponse(content={'message': 'call_already_happened'}, status_code=400)

            # ==========================================================================================================
            # REMOVE SCHEDULED JOB
            # ==========================================================================================================
            try:
                scheduler.remove_job(job_id)
                c.execute("UPDATE calls SET call_job_id=NULL, call_date=NULL, call_time=NULL WHERE user_id=?", (user_id,))
                conn.commit()
                return JSONResponse(content={'message': 'call_cancelled_successfully'}, status_code=200)
            except Exception as e:
                return JSONResponse(content={'message': 'call_not_found'}, status_code=404)
        else:
            return JSONResponse(content={'message': 'user_not_found'}, status_code=404)

    except sqlite3.Error as e:
        print(f"Database error: {e}")
        return JSONResponse(content={'message': 'error500'}, status_code=500)
    finally:
        conn.close()


# ==========================================================================================================
# CALL INITIATION FUNCTION
# ==========================================================================================================
def initiate_call(user_id):
    """
    Initiates a phone call by running the caller.py script.

    This function is called by the APScheduler when a scheduled call
    time is reached. It executes caller.py as a subprocess with the
    user_id as an argument.

    Args:
        user_id: The ID of the user to call

    Why subprocess?
    - Isolates the call initiation from the main server
    - Allows caller.py to handle Twilio API interaction separately
    - Prevents blocking the main application
    """
    print(f"initiate_call with user_id: {user_id}")

    # Build path to caller.py relative to this script
    dir_of_current_script = os.path.dirname(__file__)
    relative_path_to_caller = os.path.join(dir_of_current_script, 'caller.py')

    # Execute caller.py with user_id as argument
    subprocess.run(["python", relative_path_to_caller, str(user_id)])


# ==========================================================================================================
# PENDING CALLS SCHEDULER
# ==========================================================================================================
@app.get("/schedule-pending-calls")
async def schedule_pending_calls():
    """
    Reschedules all pending calls from the database.

    This endpoint is called at application startup to restore any
    scheduled calls that may have been lost due to server restart.

    Flow:
    1. Query all calls from database
    2. Filter out calls without date/time
    3. Validate date and time formats
    4. Schedule future calls
    5. Skip past calls

    This ensures no scheduled calls are lost if the server restarts.
    """
    conn = get_db_connection()
    c = conn.cursor()
    c.row_factory = sqlite3.Row
    c.execute("SELECT user_id, call_date, call_time, time_zone FROM calls")
    calls = c.fetchall()

    for call in calls:
        # Skip calls without schedule data
        if call['call_date'] is None or call['call_time'] is None:
            continue

        # ==========================================================================================================
        # VALIDATE DATE AND TIME FORMATS
        # ==========================================================================================================
        # Regex for YYYY-MM-DD
        date_pattern = re.compile(r'^\d{4}-\d{2}-\d{2}$')
        # Regex for HH:MM
        time_pattern = re.compile(r'^\d{2}:\d{2}$')

        if date_pattern.match(call['call_date']) and time_pattern.match(call['call_time']):
            user_id = call['user_id']
            tz = timezone(call['time_zone'])

            # Parse and localize datetime
            call_datetime_str = f"{call['call_date']} {call['call_time']}"
            call_datetime = datetime.strptime(call_datetime_str, '%Y-%m-%d %H:%M')
            call_datetime = tz.localize(call_datetime)

            # ==========================================================================================================
            # SCHEDULE ONLY FUTURE CALLS
            # ==========================================================================================================
            if call_datetime > datetime.now(tz):
                await schedule_call(None, user_id, call['call_date'], call['call_time'], call['time_zone'], conn, close_conn=False)

    print("Pending calls scheduled")


# ==========================================================================================================
# HTTP CONNECTION POOL CREATION
# ==========================================================================================================
async def create_pool():
    """
    Creates an aiohttp connection pool for API requests.

    This function creates a TCP connector and client session that will
    be used for all HTTP requests during a call (LLM API, TTS API).

    Returns:
        tuple: (connector, session)
            - connector: TCPConnector with connection pool settings
            - session: ClientSession configured with the connector

    Pool Configuration:
    - limit: Maximum concurrent connections (based on CPU count)
    - ttl_dns_cache: DNS cache lifetime (5 minutes)
    - keepalive_timeout: Keep connections alive for 30 seconds
    - connect timeout: 1 second
    - sock_connect timeout: 3 seconds

    Why connection pooling?
    - Reuses connections instead of creating new ones
    - Reduces latency from connection establishment
    - More efficient use of system resources
    """
    # Create timeout settings
    timeout = aiohttp.ClientTimeout(connect=1, sock_connect=3)

    # Create connection pool
    connector = aiohttp.TCPConnector(
        limit=max_pool,           # Max concurrent connections
        ttl_dns_cache=300,        # DNS cache for 5 minutes
        keepalive_timeout=30      # Keep connections alive 30 seconds
    )

    # Create client session with connector
    session = aiohttp.ClientSession(connector=connector, timeout=timeout)

    return connector, session


# ==========================================================================================================
# APPLICATION STARTUP
# ==========================================================================================================
async def main():
    """
    Application initialization function.

    This async function is called at startup to:
    1. Schedule any pending calls from the database
    2. Load audio files into memory

    These operations are done before the server starts accepting
    requests to ensure everything is ready.
    """
    print("------")
    print("Scheduling pending Calls..")
    await schedule_pending_calls()
    print("------")
    await load_mp3_files()


# ==========================================================================================================
# MAIN ENTRY POINT
# ==========================================================================================================
# This block runs when the script is executed directly (not imported)
if __name__ == '__main__':
    # Run the async initialization function
    asyncio.run(main())

    # ==========================================================================================================
    # DISPLAY SANTA ASCII ART
    # ==========================================================================================================
    # Fun ASCII art displayed on server startup
    print("                    _...")
    print("              o_.-\"`    `\\ ")
    print("       .--.  _ `'-._.-'\"\"-;     _")
    print("     .'    `\\_\\_  {_.-a\"a-}  _ / \\ ")
    print("   _/     .-'  '. {c-._o_.){\\|`  |")
    print("  (@`-._ /       \\{    ^  } \\\\ _/")
    print("   `~\\  '-._      /'.     }  \\}  .-.")
    print("     |>:<   '-.__/   '._,} \\_/  / ())")
    print("     |     >:<   `'---. ____'-.|(`\"\"")
    print("     \\            >:<  \\\\_\\\\_\\ | ;")
    print("      \\                 \\\\-{}-\\/  \\")
    print("       \\                 '._\\\\'   /)")
    print("        '.                       /(/")
    print("          `-._ _____ _ _____ __.'\\ \\ ")
    print("            / \\     / \\     / \\   \\ \\ ")
    print("         _.'/^\\'._.'/^\\'._.'/^\\'.__) \\ ")
    print("     ,=='  `---`   '---'   '---'      )")
    print("     `\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"\"`")
    print("***********************************************")
    print("*      Santa Claus AI loaded and ready!       *")
    print("*      Use caller.py to make calls            *")
    print("***********************************************")

    # ==========================================================================================================
    # REGISTER CLEANUP HANDLER
    # ==========================================================================================================
    # Ensure scheduler is properly shut down when application exits
    atexit.register(lambda: scheduler.shutdown())

    import uvicorn

    # ==========================================================================================================
    # SERVER FUNCTIONS
    # ==========================================================================================================
    # Define functions to run HTTPS and HTTP servers

    def run_https():
        """Run HTTPS server on port 7777 with SSL certificates."""
        uvicorn.run(
            app,
            host='0.0.0.0',
            port=7777,
            ssl_certfile='static/sec/cert.pem',
            ssl_keyfile='static/sec/privkey.pem'
        )

    def run_http():
        """Run HTTP server on port 7778 (for internal use)."""
        uvicorn.run(app, host='0.0.0.0', port=7778)

    # ==========================================================================================================
    # START SERVERS
    # ==========================================================================================================
    # Start HTTP server in background thread (for internal communication)
    http_thread = threading.Thread(target=run_http, daemon=True)
    http_thread.start()
    print("HTTP server started on port 7778")

    # Run HTTPS server in main thread (for Twilio webhooks)
    print("HTTPS server starting on port 7777")
    run_https()
