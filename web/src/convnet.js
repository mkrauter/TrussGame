// Runtime for the node detector: convolutions, in the browser, with no
// dependency and no GPU.
//
// Batch norm is folded into the preceding convolution at export time, so a
// layer here is just conv + bias + activation. That removes a full pass over
// every feature map and makes this file short enough to audit.
//
// The loop order is the whole performance story. The innermost loop holds one
// weight constant and walks a row of the input and a row of the output, both
// stride 1, which is the only shape a JS engine reliably keeps in registers.
// Written the obvious way -- output pixel outermost, kernel innermost -- this
// is several times slower.

function decode(b64) {
  const binary = atob(b64);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
  return new Float32Array(bytes.buffer);
}

function conv2d(op, input, inW, inH) {
  const { cin, cout, k, stride, dilation, pad, weight, bias } = op;
  const outW = Math.floor((inW + 2 * pad - dilation * (k - 1) - 1) / stride) + 1;
  const outH = Math.floor((inH + 2 * pad - dilation * (k - 1) - 1) / stride) + 1;
  const out = new Float32Array(cout * outH * outW);

  for (let oc = 0; oc < cout; oc++) {
    const plane = oc * outH * outW;
    out.fill(bias[oc], plane, plane + outH * outW);

    for (let ic = 0; ic < cin; ic++) {
      const inPlane = ic * inH * inW;
      const wBase = (oc * cin + ic) * k * k;

      for (let ky = 0; ky < k; ky++) {
        for (let kx = 0; kx < k; kx++) {
          const w = weight[wBase + ky * k + kx];
          if (w === 0) continue;

          for (let oy = 0; oy < outH; oy++) {
            const iy = oy * stride - pad + ky * dilation;
            if (iy < 0 || iy >= inH) continue;
            const inRow = inPlane + iy * inW;
            const outRow = plane + oy * outW;

            // Clip the x range once instead of testing every pixel.
            const startX = Math.max(0, Math.ceil((pad - kx * dilation) / stride));
            const endX = Math.min(outW, Math.ceil((inW + pad - kx * dilation) / stride));
            for (let ox = startX; ox < endX; ox++) {
              out[outRow + ox] += w * input[inRow + ox * stride - pad + kx * dilation];
            }
          }
        }
      }
    }
  }
  return { data: out, width: outW, height: outH };
}

function silu(x) {
  for (let i = 0; i < x.length; i++) x[i] = x[i] / (1 + Math.exp(-x[i]));
  return x;
}

export function decodeOps(ops) {
  return ops.map((op) =>
    op.type === 'conv'
      ? { ...op, weight: decode(op.weight), bias: decode(op.bias) }
      : op
  );
}

export function runOps(ops, input, width, height) {
  let current = { data: input, width, height };
  for (const op of ops) {
    if (op.type === 'conv') {
      current = conv2d(op, current.data, current.width, current.height);
    } else if (op.type === 'silu') {
      silu(current.data);
    } else {
      throw new Error(`unknown op ${op.type}`);
    }
  }
  return current;
}
