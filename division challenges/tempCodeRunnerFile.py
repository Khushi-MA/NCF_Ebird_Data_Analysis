    # for idx, url in enumerate(urls, start=1):
    #     print(f"\n🔗 Processing URL {idx}: {url}")
    #     try:
    #         page_content = extract_main_content(url)
    #         response = send_to_gemini(API_KEY, page_content)
    #         response = extract_clean_gemini_json(response)

    #         print("📥 Gemini Response:", response)
    #         # pretty_print_gemini_response(response)
    #     except Exception as e:
    #         print(f"❌ Failed to process URL {idx}: {e}")