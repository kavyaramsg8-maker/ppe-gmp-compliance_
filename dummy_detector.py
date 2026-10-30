def detect(frame_content, camera_id, session_id):
    """
    Dummy detector for testing.
    Accepts the 3 arguments Kavya's worker passes in.
    """
    print(f"✅ Dummy detector processed a frame for {camera_id}")
    
    # The worker expects a list of detected persons. 
    # Returning an empty list means "no violations found" (COMPLIANT).
    return []