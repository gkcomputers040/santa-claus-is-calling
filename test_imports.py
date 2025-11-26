"""
Test script to verify all required dependencies are installed correctly.
This script attempts to import all major dependencies used in the project.
"""

import sys

def test_imports():
    """Test importing all required packages"""
    errors = []
    success_count = 0

    # List of all imports used in the project
    imports_to_test = [
        # Core frameworks
        ('flask', 'Flask'),
        ('fastapi', 'FastAPI'),
        ('uvicorn', 'Uvicorn'),
        ('starlette', 'Starlette'),

        # Async & HTTP
        ('aiohttp', 'aiohttp'),
        ('aiofiles', 'aiofiles'),
        ('websockets', 'websockets'),
        ('requests', 'requests'),

        # AI/ML APIs
        ('openai', 'OpenAI'),
        ('deepgram', 'Deepgram SDK'),

        # Twilio
        ('twilio', 'Twilio'),

        # Data processing
        ('numpy', 'NumPy'),
        ('scipy', 'SciPy'),
        ('orjson', 'orjson'),

        # Audio
        ('pydub', 'pydub'),

        # Scheduling
        ('apscheduler', 'APScheduler'),

        # Security & Database
        ('bcrypt', 'bcrypt'),
        ('dotenv', 'python-dotenv'),
        ('itsdangerous', 'itsdangerous'),
        ('markupsafe', 'MarkupSafe'),

        # Data validation
        ('pydantic', 'Pydantic'),

        # Date & Time
        ('pytz', 'pytz'),
        ('dateutil', 'python-dateutil'),

        # Utilities
        ('emoji', 'emoji'),
        ('psutil', 'psutil'),
    ]

    print("Testing imports...")
    print("-" * 50)

    for module_name, display_name in imports_to_test:
        try:
            __import__(module_name)
            print(f"✓ {display_name:20} - OK")
            success_count += 1
        except ImportError as e:
            error_msg = f"✗ {display_name:20} - FAILED: {str(e)}"
            print(error_msg)
            errors.append(error_msg)

    print("-" * 50)
    print(f"\nResults: {success_count}/{len(imports_to_test)} imports successful")

    if errors:
        print("\n❌ FAILED - The following imports failed:")
        for error in errors:
            print(f"  {error}")
        return False
    else:
        print("\n✅ SUCCESS - All imports working correctly!")
        return True

if __name__ == "__main__":
    success = test_imports()
    sys.exit(0 if success else 1)
