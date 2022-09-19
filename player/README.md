# player

## Tools and Frameworks we use

1. dash.js - For adaptive bitrate streaming via DASH.
2. three.js - For 3D rendering, with CSS3DRenderer.js so far (WebXR not supported).
3. aframe - For 3D rendering (WebXR supported for any videos based on VP9).
4. angular.js - For data virtualization and code optimization.
5. sensitive segmentation suites - For video's content analystics.


## How to run
#### Player over aframe

1. Run the HTML file via HTTP address (e.g., http://localhost/vr-dash-tile-player/Index.html).
2. Confirm the location of JSON file, the Mode (VOD/LIVE) and the Rule(FOVRule/HighestBitrateRule/FOVContentRule/ThroughputRule) you want to use in HTML page, then click "link".
3. Click "aframe" to load aframe page.
4. Click "load" to initialize MediaPlayer according to JSON description.
5. Click "Play" and "Pause" to control the player.
