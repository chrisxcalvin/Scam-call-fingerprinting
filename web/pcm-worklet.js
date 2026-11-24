// AudioWorklet: forwards mono Float32 blocks (128 samples at the context rate, 16 kHz) to the main thread.
class PCMForwarder extends AudioWorkletProcessor {
  process(inputs) {
    const ch = inputs[0] && inputs[0][0];
    if (ch) this.port.postMessage(ch.slice(0));
    return true;
  }
}
registerProcessor("pcm-forwarder", PCMForwarder);
