# Workshop GIFs and Animations Directory

This directory contains animated GIFs and videos for demonstration purposes in your workshop.

## Recommended Animation Files

### Live Demos:
- `demo-walkthrough.gif` - Step-by-step demo process
- `feature-showcase.gif` - Key feature demonstrations
- `before-after-animation.gif` - Transformation visualizations

### UI Interactions:
- `interface-navigation.gif` - How to navigate your tool/interface
- `code-execution.gif` - Code running and producing output
- `installation-process.gif` - Setup and installation steps

### Process Visualizations:
- `algorithm-visualization.gif` - How your algorithm works
- `data-flow.gif` - Data processing pipeline
- `workflow-animation.gif` - Complete workflow demonstration

## Creating GIFs

### From Screen Recordings:

#### macOS:
```bash
# Record with QuickTime, then convert
ffmpeg -i recording.mov -vf "fps=10,scale=800:-1:flags=lanczos" -c:v gif demo.gif
```

#### Linux:
```bash
# Using byzanz
byzanz-record --duration=30 --x=0 --y=0 --width=800 --height=600 demo.gif

# Using ffmpeg from screen
ffmpeg -f x11grab -s 1280x720 -i :0.0 -vf "fps=10,scale=800:-1:flags=lanczos" demo.gif
```

#### Windows:
```bash
# Using ffmpeg with GDI capture
ffmpeg -f gdigrab -i desktop -vf "fps=10,scale=800:-1:flags=lanczos" demo.gif
```

### Optimization:
```bash
# Reduce file size while maintaining quality
gifsicle --optimize=3 --resize-width 800 input.gif > output.gif

# Further compression
ffmpeg -i input.gif -vf "fps=8,scale=640:-1:flags=lanczos,palettegen" palette.png
ffmpeg -i input.gif -i palette.png -filter_complex "fps=8,scale=640:-1:flags=lanczos[x];[x][1:v]paletteuse" optimized.gif
```

## GIF Guidelines

### Technical Specifications:
- **Resolution**: 800px width maximum (for web performance)
- **Frame rate**: 8-12 FPS (balance between smoothness and file size)
- **Duration**: 5-30 seconds (longer animations should be videos)
- **File size**: Under 10MB per GIF (preferably under 5MB)

### Quality Guidelines:
- Focus on the important part of the screen
- Use consistent timing between frames
- Include smooth transitions
- Crop unnecessary UI elements
- Consider adding subtle highlights or callouts

### Design Principles:
- Show, don't tell - demonstrate actual functionality
- Keep animations focused on one concept
- Use consistent visual style across GIFs
- Ensure readability at presentation scale
- Consider adding text overlays for clarity

## Alternative Formats

For larger or higher-quality demonstrations:

### WebM Videos:
```bash
# Convert to WebM for better compression
ffmpeg -i demo.gif -c:v libvpx-vp9 -crf 30 -b:v 0 demo.webm
```

### MP4 Videos:
```bash
# Convert to MP4 for broad compatibility
ffmpeg -i demo.gif -c:v libx264 -pix_fmt yuv420p demo.mp4
```

### HTML5 Video in Slides:
```html
<video width="800" height="600" autoplay loop muted>
  <source src="./gifs/demo.webm" type="video/webm">
  <source src="./gifs/demo.mp4" type="video/mp4">
  <img src="./gifs/demo.gif" alt="Demo fallback">
</video>
```

## Usage in Slides

Embed GIFs in your presentation:

```markdown
![Demo Walkthrough](./gifs/demo-walkthrough.gif)

<div style="text-align: center;">
<img src="./gifs/feature-showcase.gif" alt="Feature Demo" width="600">
</div>

<!-- For looping control -->
<img src="./gifs/process-animation.gif" alt="Process Flow" width="800" style="border: 2px solid #ddd;">
```

## Best Practices

### Recording Tips:
1. **Clean environment**: Close unnecessary applications
2. **Consistent speed**: Move mouse and type at steady pace
3. **Clear actions**: Make deliberate, visible interactions
4. **Multiple takes**: Record several versions and choose the best
5. **Script actions**: Plan your demonstration sequence

### Post-processing:
1. **Trim carefully**: Remove dead time at start/end
2. **Add highlights**: Circle or highlight important areas
3. **Consistent branding**: Use consistent colors/styling
4. **Test playback**: Verify GIFs play correctly in browsers

### Performance Considerations:
- Load GIFs lazily in web presentations
- Provide static fallback images
- Consider user bandwidth limitations
- Test on different devices and connections

## Tools and Resources

### Recording Software:
- **LICEcap** (Free, cross-platform)
- **Kap** (macOS, open source)
- **ScreenToGif** (Windows, open source)
- **Byzanz** (Linux)

### Optimization Tools:
- **gifsicle** (Command line, cross-platform)
- **ezgif.com** (Online optimization)
- **GIMP** (Full-featured editor)
