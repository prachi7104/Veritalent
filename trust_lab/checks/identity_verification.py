def check_identity_verification(candidate: dict) -> dict:
    """
    Composite of verified_email, verified_phone, and linkedin_connected.
    Returns an unverified fraction (0.0 = all verified, 1.0 = none verified).
    """
    signals = candidate.get("redrob_signals", {})
    
    email_ver = bool(signals.get("verified_email", False))
    phone_ver = bool(signals.get("verified_phone", False))
    li_conn = bool(signals.get("linkedin_connected", False))
    
    total = 3
    verified = sum([email_ver, phone_ver, li_conn])
    
    unverified_fraction = (total - verified) / total
    
    return {
        "unverified_fraction": float(unverified_fraction),
        "details": f"Email: {email_ver}, Phone: {phone_ver}, LinkedIn: {li_conn}"
    }
