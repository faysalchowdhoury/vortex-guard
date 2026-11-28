import numpy as np, torch
from vortexguard.models import VortexGuard, StreamingVortexGuard, flux_features, FluxState
from vortexguard.tokenizer import encode_adfa, x86_64_to_token, I386, OFFSET, X86_64

def test_streaming_matches_batch():
    torch.manual_seed(0)
    m = VortexGuard().eval()
    seq = encode_adfa(np.random.RandomState(0).choice([3,4,5,6,102,120,11,213], 64)).tok
    flux = torch.from_numpy(flux_features(seq)).unsqueeze(0)
    tok = torch.from_numpy(seq).unsqueeze(0)
    with torch.no_grad():
        full = float(m(tok, flux, torch.ones_like(tok))["logit"])
    s = StreamingVortexGuard(m)
    for t in seq: last = s.step(int(t))
    assert abs(full - last) < 1e-4, (full, last)

def test_flux_streaming():
    seq = encode_adfa([3,4,102,120,213]*20).tok
    f = flux_features(seq); st = FluxState()
    g = np.stack([st.step(int(t)) for t in seq])
    assert np.allclose(f, g, atol=1e-6)

def test_live_translation():
    assert x86_64_to_token(X86_64["read"]) == I386["read"] + OFFSET
    assert x86_64_to_token(X86_64["newfstatat"]) == I386["fstatat64"] + OFFSET
