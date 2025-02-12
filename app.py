import streamlit as st
from youtube_transcript_api import YouTubeTranscriptApi
from transformers import pipeline
import requests
import json
from youtube_transcript_api._errors import NoTranscriptFound, TranscriptsDisabled
import time
import re

# Configure page
st.set_page_config(
    page_title="YouTube Video Analyzer",
    page_icon="🎥",
    layout="wide"
)

@st.cache_resource
def load_model():
    """Load the summarization model"""
    try:
        # Using BART model for summarization
        summarizer = pipeline(
            "summarization",
            model="facebook/bart-large-cnn",
            device=-1  # CPU
        )
        return summarizer
    except Exception as e:
        st.error(f"Failed to load model: {str(e)}")
        return None

def generate_summary(transcript, summarizer):
    """Generate summary using HuggingFace model"""
    try:
        # Split long transcripts into chunks if needed
        max_chunk_length = 1024
        chunks = [transcript[i:i + max_chunk_length] for i in range(0, len(transcript), max_chunk_length)]
        
        summaries = []
        for chunk in chunks:
            # Calculate dynamic max_length based on input length
            input_length = len(chunk.split())
            max_length = min(300, max(50, input_length // 2))  # Set max_length to half of input length
            min_length = max(30, input_length // 4)  # Set min_length to quarter of input length
            
            summary = summarizer(chunk, 
                               max_length=max_length, 
                               min_length=min_length, 
                               do_sample=False)
            summaries.append(summary[0]['summary_text'])
        
        return " ".join(summaries)
    except Exception as e:
        st.error(f"Error generating summary: {str(e)}")
        return None

def process_video_content(transcript, timestamps):
    """Process video content using HuggingFace model"""
    try:
        summarizer = load_model()
        if not summarizer:
            return None

        summary = generate_summary(transcript, summarizer)
        if not summary:
            return None

        # Calculate section lengths based on summary length
        summary_length = len(summary)
        topic_length = min(200, summary_length // 4)
        insight_length = min(100, summary_length // 5)

        # Split summary into sentences for better formatting
        sentences = summary.split('. ')
        
        # Format main topics (first 3-4 sentences as bullet points)
        main_topics = '\n'.join([f"• {sent.strip()}" for sent in sentences[:3]])
        
        # Format detailed summary (remaining sentences in paragraphs)
        detailed_points = sentences[3:]
        paragraphs = []
        current_para = []
        
        for i, sent in enumerate(detailed_points):
            current_para.append(sent)
            if len(current_para) == 3 or i == len(detailed_points) - 1:  # Create paragraph every 3 sentences
                paragraphs.append('. '.join(current_para) + '.')
                current_para = []
        
        detailed_summary = '\n\n'.join(paragraphs)
        
        # Extract key phrases for insights and technical details
        key_phrases = summary.split(',')
        insights = [phrase.strip() for phrase in key_phrases[:4]]
        tech_details = [phrase.strip() for phrase in key_phrases[4:7]]

        # Format the summary according to our template
        formatted_summary = f"""
[MAIN TOPICS]
{main_topics}

[DETAILED SUMMARY]
{detailed_summary}

[KEY INSIGHTS]
• {insights[0] if len(insights) > 0 else 'Key insight 1'}
• {insights[1] if len(insights) > 1 else 'Key insight 2'}
• {insights[2] if len(insights) > 2 else 'Key insight 3'}

[TECHNICAL DETAILS]
• {tech_details[0] if len(tech_details) > 0 else 'Technical aspect 1'}
• {tech_details[1] if len(tech_details) > 1 else 'Technical aspect 2'}
• {tech_details[2] if len(tech_details) > 2 else 'Technical aspect 3'}

[TIMESTAMPS]
{timestamps[:5] if timestamps else 'Important moments from the video.'}
"""
        return formatted_summary

    except Exception as e:
        st.error(f"Error processing content: {str(e)}")
        return None

def extract_transcript_details(youtube_video_url):
    """Extract and process transcript from YouTube video"""
    try:
        # List of possible URL patterns and their video ID extraction methods
        url_patterns = [
            # Standard watch URLs
            r'(?:youtube\.com/watch\?v=|youtu\.be/|youtube\.com/embed/|youtube\.com/v/|youtube\.com/e/)([^&?#]+)',
            # Short URLs
            r'youtube\.com/shorts/([^&?#]+)',
            # Live URLs
            r'youtube\.com/live/([^&?#]+)',
            # Attribution links
            r'youtube\.com/attribution_link.*watch%3Fv%3D([^%&]+)',
            # Plain video ID in path
            r'youtube\.com/watch/([^&?#]+)',
            # Mobile URLs
            r'm\.youtube\.com/watch\?v=([^&?#]+)',
        ]
        
        video_id = None
        for pattern in url_patterns:
            match = re.search(pattern, youtube_video_url)
            if match:
                video_id = match.group(1)
                break
                
        if not video_id:
            st.error("""
            Invalid YouTube URL format. Supported formats include:
            - Standard watch URLs (youtube.com/watch?v=...)
            - Short URLs (youtu.be/...)
            - Embedded URLs (youtube.com/embed/...)
            - Short-form videos (youtube.com/shorts/...)
            - Live streams (youtube.com/live/...)
            """)
            return None, None
            
        # Add longer delay for Hugging Face Spaces
        time.sleep(3)
        
        try:
            # Direct transcript fetch first
            transcript_text = YouTubeTranscriptApi.get_transcript(video_id, languages=['en'])
            
        except Exception as first_error:
            try:
                # Second attempt with list_transcripts
                transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
                transcript = transcript_list.find_transcript(['en'])
                transcript_text = transcript.fetch()
                
            except Exception as second_error:
                try:
                    # Third attempt with auto-generated
                    transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
                    transcript = transcript_list.find_generated_transcript(['en'])
                    transcript_text = transcript.fetch()
                    
                except Exception as e:
                    st.error(f"""
                    Unable to access video transcripts. This might be due to:
                    1. Video restrictions in Hugging Face Spaces
                    2. YouTube API limitations
                    
                    Please try:
                    1. A different YouTube video
                    2. A video with manual English captions
                    3. Running the app locally
                    
                    Error details: {str(e)}
                    """)
                    return None, None

        # Process transcript with timestamps
        full_transcript = ""
        timestamps = []
        
        for entry in transcript_text:
            time_value = int(entry['start'])
            text = entry['text']
            minutes = time_value // 60
            seconds = time_value % 60
            timestamp = f"{minutes:02d}:{seconds:02d}"
            
            full_transcript += f" {text}"
            if len(text.split()) > 5:
                timestamps.append(f"{timestamp} - {text}")

        if not full_transcript.strip():
            st.error("Could not extract meaningful text from the video.")
            return None, None

        st.success(f"✅ Successfully extracted transcript ({len(full_transcript.split())} words)")
        return full_transcript, timestamps

    except Exception as e:
        st.error(f"""
        Error accessing video: {str(e)}
        
        When using Hugging Face Spaces, try:
        1. Videos with manual English captions
        2. Popular videos with verified captions
        3. Shorter videos
        4. Running the app locally for full functionality
        """)
        return None, None

def generate_ollama_summary(transcript, timestamps):
    """Generate summary using local Llama 2 model"""
    try:    
        # Prepare prompt with transcript and timestamps
        PROMPT_TEMPLATE = """Please analyze this video transcript and provide:
1. Main topics covered
2. Detailed summary
3. Key insights and takeaways
4. Technical details mentioned
5. Important timestamps

Transcript: {transcript}"""
        
        formatted_prompt = PROMPT_TEMPLATE.format(
            transcript=transcript
        )

        # Call local Ollama API with Llama 2
        response = requests.post(
            'http://localhost:11434/api/generate',
            json={
                'model': 'llama3.2:latest',  # Using Llama 2 70B model
                'prompt': formatted_prompt,
                'stream': False,
                'options': {
                    'temperature': 0.3,  # Lower temperature for more focused output
                    'top_p': 0.7,
                    'top_k': 50,
                    'num_ctx': 4096,  # Larger context window
                    'num_predict': 1024  # Longer response length
                }
            }
        )
        
        if response.status_code == 200:
            return response.json()['response']
        else:
            st.error("Failed to get response from Ollama")
            return None

    except Exception as e:
        st.error(f"Error generating summary: {str(e)}")
        return None

# Streamlit UI
st.title("🎥 YouTube Video Analysis Tool")
st.markdown("### Get detailed insights from any YouTube video")

# Add this before the input section
st.markdown("""
### Tips for best results:
- Use videos with English subtitles
- Ensure the video is public and not age-restricted
- Try shorter videos first
- If you get an error, wait a minute and try again
""")

# Input section
youtube_link = st.text_input("Enter YouTube Video Link:", 
                            placeholder="https://www.youtube.com/watch?v=...")

if youtube_link:
    try:
        video_id = youtube_link.split("=")[1]
        # Display video thumbnail with width set to match container
        col1, col2, col3 = st.columns([1,6,1])
        with col2:
            st.image(f"http://img.youtube.com/vi/{video_id}/0.jpg")
        
        if st.button("Analyze Video", type="primary"):
            with st.spinner("Analyzing video content..."):
                # Get transcript and timestamps
                transcript, timestamps = extract_transcript_details(youtube_link)
                
                if transcript:
                    # Generate summary
                    summary = process_video_content(transcript, timestamps)
                    
                    if summary:
                        # Display results in an organized layout
                        col1, col2 = st.columns([2, 1])
                        
                        with col1:
                            st.markdown("## 📝 Detailed Analysis")
                            st.markdown(summary)
                        
                        with col2:
                            st.markdown("## ⏰ Key Moments")
                            if timestamps:
                                for timestamp in timestamps[:10]:  # Show first 10 timestamps
                                    st.markdown(f"• {timestamp}")
                            
                            st.markdown("## 💡 Tips")
                            st.markdown("""
                            - Use timestamps to navigate to specific sections
                            - Check technical details for accuracy
                            - Review key insights for main takeaways
                            """)

    except Exception as e:
        st.error(f"Error processing video: {str(e)}")

# Footer
st.markdown("---")
st.markdown("Made with ❤️ using Ollama and Streamlit")




