import math

def _invar_sdiv(a, b):
    if not math.isfinite(a):
        a = math.copysign(1e+100, a) if math.isinf(a) else 0.0
    if not math.isfinite(b):
        return 0.0
    if b >= 0.0 and b < 1e-30:
        return a / 1e-30
    if b < 0.0 and b > -1e-30:
        return a / -1e-30
    return a / b

def mm_rate(s, vmax, km):
    denom = km + s
    if denom <= 0.0:
        return 0.0
    return _invar_sdiv(vmax * s, denom)

def layer_1_step(buf, dt):
    s = buf.data[0]
    p = buf.data[1]
    vmax = buf.data[2]
    km = buf.data[3]
    kc = buf.data[4]
    if buf.data[5] == 0.0:
        buf.data[5] = s
        buf.data[6] = p
        buf.data[9] = s + p
    r1 = mm_rate(s, vmax, km)
    k1s = -r1
    k1p = r1 - kc * p
    s2 = s + 0.5 * dt * k1s
    p2 = p + 0.5 * dt * k1p
    if s2 < 0.0:
        s2 = 0.0
    if p2 < 0.0:
        p2 = 0.0
    r2 = mm_rate(s2, vmax, km)
    k2s = -r2
    k2p = r2 - kc * p2
    s3 = s + 0.5 * dt * k2s
    p3 = p + 0.5 * dt * k2p
    if s3 < 0.0:
        s3 = 0.0
    if p3 < 0.0:
        p3 = 0.0
    r3 = mm_rate(s3, vmax, km)
    k3s = -r3
    k3p = r3 - kc * p3
    s4 = s + dt * k3s
    p4 = p + dt * k3p
    if s4 < 0.0:
        s4 = 0.0
    if p4 < 0.0:
        p4 = 0.0
    r4 = mm_rate(s4, vmax, km)
    k4s = -r4
    k4p = r4 - kc * p4
    s_new = s + dt / 6.0 * (k1s + 2.0 * k2s + 2.0 * k3s + k4s)
    p_new = p + dt / 6.0 * (k1p + 2.0 * k2p + 2.0 * k3p + k4p)
    if s_new < 0.0:
        s_new = 0.0
    if p_new < 0.0:
        p_new = 0.0
    conv = s - s_new - (p_new - p + dt * kc * p)
    buf.data[0] = s_new
    buf.data[1] = p_new
    buf.data[7] = buf.data[7] + conv * conv
    if p_new > buf.data[8]:
        buf.data[8] = p_new
    if buf.data[0] < 0.0:
        buf.data[0] = 0.0
    if buf.data[1] < 0.0:
        buf.data[1] = 0.0
    if buf.data[8] <= 0.0:
        buf.data[8] = 0.0500000005
    if not math.isfinite(buf.data[0]) or buf.data[0] > 0.001:
        buf.data[0] = 0.001
    if not math.isfinite(buf.data[7]) or buf.data[7] > 0.0001:
        buf.data[7] = 0.0001
    if not math.isfinite(buf.data[8]) or buf.data[8] > 0.100000001:
        buf.data[8] = 0.100000001
    if not math.isfinite(buf.data[1]):
        buf.data[1] = 0.0

def layer_2_step(buf, dt):
    layer_1_step(buf, dt)
    if buf.data[25] < 0.0:
        buf.data[25] = 0.0
    if buf.data[26] < 0.0:
        buf.data[26] = 0.0
    if buf.data[8] <= 0.0:
        buf.data[8] = 0.0500000005
    if not math.isfinite(buf.data[25]) or buf.data[25] > 0.001:
        buf.data[25] = 0.001
    if not math.isfinite(buf.data[7]) or buf.data[7] > 0.0001:
        buf.data[7] = 0.0001
    if not math.isfinite(buf.data[8]) or buf.data[8] > 0.100000001:
        buf.data[8] = 0.100000001
    if not math.isfinite(buf.data[26]):
        buf.data[26] = 0.0
    if buf.data[0] < 0.0:
        buf.data[0] = 0.0
    if buf.data[1] < 0.0:
        buf.data[1] = 0.0
    if not math.isfinite(buf.data[0]) or buf.data[0] > 0.100000001:
        buf.data[0] = 0.100000001
    if not math.isfinite(buf.data[1]):
        buf.data[1] = 0.0

def layer_3_step(buf, dt):
    layer_2_step(buf, dt)
    if buf.data[29] < 0.0:
        buf.data[29] = 0.0
    if buf.data[30] < 0.0:
        buf.data[30] = 0.0
    if buf.data[26] < 0.0:
        buf.data[26] = 0.0
    if buf.data[28] < 0.0:
        buf.data[28] = 0.0
    if buf.data[8] <= 0.0:
        buf.data[8] = 0.0500000005
    if not math.isfinite(buf.data[29]) or buf.data[29] > 0.100000001:
        buf.data[29] = 0.100000001
    if not math.isfinite(buf.data[7]) or buf.data[7] > 0.0001:
        buf.data[7] = 0.0001
    if not math.isfinite(buf.data[8]) or buf.data[8] > 0.100000001:
        buf.data[8] = 0.100000001
    if not math.isfinite(buf.data[25]) or buf.data[25] > 0.001:
        buf.data[25] = 0.001
    if not math.isfinite(buf.data[30]):
        buf.data[30] = 0.0
    if not math.isfinite(buf.data[28]):
        buf.data[28] = 0.0
    if buf.data[0] < 0.0:
        buf.data[0] = 0.0
    if buf.data[1] < 0.0:
        buf.data[1] = 0.0
    if not math.isfinite(buf.data[0]) or buf.data[0] > 0.100000001:
        buf.data[0] = 0.100000001
    if not math.isfinite(buf.data[1]):
        buf.data[1] = 0.0

def simulation_step(buf, dt):
    layer_3_step(buf, dt)
