---
layout: default
title: Results
---

Inference Arena runs the same training and inference workload through every
supported ML framework on every available platform, then publishes the
results side by side. Each tab below covers one model — pick your model,
filter to the frameworks you care about, and compare. Lower numbers are
better; **bold** marks the best matching framework on each platform.

<div class="tabs">
  <input type="radio" name="model" id="tab-smollm2" value="panel-smollm2" checked>
  <label for="tab-smollm2">SmolLM2-135M</label>
  <input type="radio" name="model" id="tab-smolvla" value="panel-smolvla">
  <label for="tab-smolvla">SmolVLA</label>
  <input type="radio" name="model" id="tab-sd" value="panel-sd">
  <label for="tab-sd">Stable Diffusion</label>
  <input type="radio" name="model" id="tab-resnet" value="panel-resnet">
  <label for="tab-resnet">ResNet-50</label>
  <input type="radio" name="model" id="tab-whisper" value="panel-whisper">
  <label for="tab-whisper">Whisper-tiny</label>
</div>

<div id="panel-smollm2" class="tab-panel active" markdown="1">

{% include smollm2-135m.md %}

</div>

<div id="panel-smolvla" class="tab-panel" markdown="1">

{% include smolvla.md %}

</div>

<div id="panel-sd" class="tab-panel" markdown="1">

{% include stablediffusion.md %}

</div>

<div id="panel-resnet" class="tab-panel" markdown="1">

{% include resnet-50.md %}

</div>

<div id="panel-whisper" class="tab-panel" markdown="1">

{% include whisper-tiny.md %}

</div>
