first_name = "original_image.dzi"
second_name = "slide.dzi" 

def get_html_file(mode):
    if mode == 1:
        name = first_name
    else:
        name = second_name

    return f'''
    <!DOCTYPE html>
    <html lang="en">
    <head>
    <meta charset="UTF-8">
    <title>original image Viewer</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/openseadragon/4.0.0/openseadragon.min.js"></script>
    <style>
        body, html {{
        margin: 0;
        padding: 0;
        overflow: hidden;
        }}

        #container {{
        position: relative;
        width: 100%;
        height: 100vh;
        }}

        #openseadragon {{
        width: 100%;
        height: 100%;
        z-index: 1;
        }}

        #drawCanvas {{
        position: absolute;
        top: 0;
        left: 0;
        z-index: 2;
        width: 100%;
        height: 100%;
        pointer-events: none;
        }}
    </style>
    </head>
    <body>
    <div id="container">
        <div id="openseadragon"></div>
        <canvas id="drawCanvas"></canvas>
    </div>

    <script>
        window.onload = function () {{
        var viewer = OpenSeadragon({{
            id: "openseadragon",
            prefixUrl: "https://cdnjs.cloudflare.com/ajax/libs/openseadragon/4.0.0/images/",
            showNavigator: true
        }});

        viewer.addTiledImage({{
            tileSource: "{name}",
            x: 0,
            y: 0,
            width: 1
        }});

        const canvas = document.getElementById("drawCanvas");
        const ctx = canvas.getContext("2d");
        let drawing = false;

        document.getElementById("drawCanvas").addEventListener("contextmenu", e => e.preventDefault());

        function resizeCanvas() {{
            canvas.width = canvas.offsetWidth;
            canvas.height = canvas.offsetHeight;
        }}
        window.addEventListener("resize", resizeCanvas);
        resizeCanvas();

        canvas.addEventListener("mousedown", e => {{
            if (!drawingEnabled) return;
            drawing = true;
            ctx.beginPath();
            ctx.moveTo(e.offsetX, e.offsetY);
        }});

        canvas.addEventListener("mousemove", e => {{
            if (!drawing || !drawingEnabled) return;
            ctx.lineTo(e.offsetX, e.offsetY);
            ctx.strokeStyle = "red";
            ctx.lineWidth = 2;
            ctx.stroke();
        }});

        canvas.addEventListener("mouseup", () => {{
            drawing = false;
        }});

        let drawingEnabled = false;

        window.toggleDrawingMode = function () {{
            drawingEnabled = !drawingEnabled;
            canvas.style.pointerEvents = drawingEnabled ? "auto" : "none";

            if (drawingEnabled) {{
            viewer.setMouseNavEnabled(false);
            viewer.innerTracker.setTracking(false);
            }} else {{
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            viewer.setMouseNavEnabled(true);
            viewer.innerTracker.setTracking(true);
            }}
        }};

        // Also keep other sync functions exposed
        window.getZoomLevel = function () {{
            return viewer.viewport.getZoom();
        }};

        window.getCenter = function () {{
            const center = viewer.viewport.getCenter();
            return {{ x: center.x, y: center.y }};
        }};

        window.syncTo = function (zoom, x, y) {{
            viewer.viewport.zoomTo(zoom);
            viewer.viewport.panTo(new OpenSeadragon.Point(x, y));
        }};
        }};
    </script>
    </body>
    </html>
    '''
