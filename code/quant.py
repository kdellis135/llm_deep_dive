"""Weight quantization basics: absmax int8, asymmetric (zero-point), and group-wise int4."""
import torch


# --8<-- [start:quant]
def quantize_absmax_int8(w):
    """Symmetric: q = round(w / s), s = max|w| / 127.  Dequantize: w ≈ s * q."""
    s = w.abs().max() / 127.0
    q = torch.clamp(torch.round(w / s), -127, 127).to(torch.int8)
    return q, s


def quantize_zeropoint(w, bits=8):
    """Asymmetric: maps [min, max] onto [0, 2^bits - 1] with a zero-point offset."""
    qmax = 2 ** bits - 1
    s = (w.max() - w.min()) / qmax
    z = torch.round(-w.min() / s)
    q = torch.clamp(torch.round(w / s) + z, 0, qmax)
    return q, s, z                     # dequantize: s * (q - z)


def quantize_groupwise(w, bits=4, group=128):
    """Symmetric quantization with one scale per `group` consecutive weights in a row.
    Small groups isolate outliers, which is why int4 schemes (GPTQ/AWQ/QLoRA) use them."""
    d_out, d_in = w.shape
    qmax = 2 ** (bits - 1) - 1
    wg = w.reshape(d_out, d_in // group, group)
    s = wg.abs().amax(dim=-1, keepdim=True) / qmax
    q = torch.clamp(torch.round(wg / s), -qmax - 1, qmax)
    return (q * s).reshape(d_out, d_in)            # return dequantized weights for comparison
# --8<-- [end:quant]
