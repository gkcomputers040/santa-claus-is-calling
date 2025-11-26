# Source: https://www.twilio.com/docs/verify/default-phone-verification-languages

languages_for_phone_number = {
    "54": "es",   # Argentina
    "43": "de",   # Austria
    "32": "fr",   # Belgium
    "591": "es",  # Bolivia
    "55": "pt",   # Brazil
    "56": "es",   # Chile
    "86": "zh-CN",# China
    "57": "es",   # Colombia
    "506": "es",  # Costa Rica
    "53": "es",   # Cuba
    "357": "el",  # Cyprus
    "1809": "es", # Dominican Republic
    "593": "es",  # Ecuador
    "20": "ar",   # Egypt
    "503": "es",  # El Salvador
    "372": "et",  # Estonia
    "33": "fr",   # France
    "49": "de",   # Germany
    "502": "es",  # Guatemala
    "504": "es",  # Honduras
    "91": "en",   # India
    "62": "id",   # Indonesia
    "39": "it",   # Italy
    "81": "ja",   # Japan
    "965": "ar",  # Kuwait
    "370": "lt",  # Lithuania
    "352": "de",  # Luxembourg
    "52": "es",   # Mexico
    "377": "fr",  # Monaco
    "505": "es",  # Nicaragua
    "968": "ar",  # Oman
    "507": "es",  # Panama
    "595": "es",  # Paraguay
    "51": "es",   # Peru
    "48": "pl",   # Poland
    "351": "pt",  # Portugal
    "226": "ro",  # Romania
    "378": "it",  # San Marino
    "966": "ar",  # Saudi Arabia
    "421": "sk",  # Slovakia
    "27": "af",   # South Africa
    "34": "es",   # Spain
    "41": "de",   # Switzerland
    "380": "uk",  # Ukraine
    "971": "ar",  # United Arab Emirates
    "44": "en-GB",# United Kingdom
    "1": "en",    # United States
    "598": "es",  # Uruguay
    "379": "it",  # Vatican City
    "58": "es",   # Venezuela
    "84": "vi"    # Vietnam
}

# Function to determine the language based on the phone number

def get_country_language(phone_number):
    # Check if the number starts with '+'
    if phone_number.startswith("+"):
        # Remove the '+' and search for the longest possible prefix in the dictionary
        just_nums = phone_number[1:]
        for i in range(1, 6):  # Assuming the longest prefix has 5 digits
            prefix = just_nums[:i]
            if prefix in languages_for_phone_number:
                return languages_for_phone_number[prefix]
        # If the prefix is not found in the dictionary
        print(f"Language not found for {phone_number}")
        return "en"
    else:
        print(f"Language not found for {phone_number}")
        return "en"
