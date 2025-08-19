# Workshop Images Directory

This directory contains all visual assets for your workshop presentation.

## Recommended Image Files

### Required Images (update filenames in slides.md and index.html):
- `hero-image.png` - Main workshop logo/illustration (400x400px recommended)
- `hero-background.jpg` - Optional background for first slide (1920x1080px)
- `architecture-diagram.png` - System overview diagram
- `solution-diagram.png` - Your solution visualization
- `thank-you-image.png` - Closing slide visual

### Optional Assets:
- `your-photo.jpg` - Presenter headshot (300x300px, circular crop recommended)
- `company-logo.png` - Organization branding
- `before-after-comparison.png` - Performance comparison visuals

## Image Guidelines

### File Formats:
- **PNG**: Use for logos, diagrams, screenshots (supports transparency)
- **JPG**: Use for photos, backgrounds (smaller file size)
- **SVG**: Use for simple graphics that need to scale (vector format)

### Size Recommendations:
- **Hero images**: 400x400px or larger
- **Background images**: 1920x1080px (16:9 aspect ratio)
- **Diagrams**: 800px wide minimum for readability
- **Profile photos**: 300x300px
- **Screenshots**: High resolution, then scale down in presentation

### Optimization:
- Compress images to reduce loading time
- Use appropriate resolution for projection (1920x1080)
- Consider using WebP format for better compression
- Keep total image directory under 50MB when possible

### Naming Convention:
- Use lowercase with hyphens: `my-diagram.png`
- Be descriptive: `performance-comparison.png` not `img1.png`
- Include version numbers for iterations: `diagram-v2.png`

## Workshop-Specific Images

Add your workshop-specific image requirements here:

```
images/
├── [workshop-topic]/          # Group related images
│   ├── demo-screenshot-1.png
│   ├── demo-screenshot-2.png
│   └── workflow-diagram.svg
├── performance/               # Performance-related visuals
│   ├── before-benchmark.png
│   ├── after-benchmark.png
│   └── comparison-chart.svg
└── examples/                  # Code example screenshots
    ├── example-1.png
    └── example-2.png
```

## Usage in Slides

Reference images in your slides.md file:

```markdown
<img src="./images/hero-image.png" alt="Workshop Hero" width="400" height="400">

![Architecture Diagram](./images/architecture-diagram.png)

<div style="text-align: center;">
<img src="./images/solution-diagram.png" alt="Solution Overview" width="700">
</div>
```

## Accessibility

- Always include meaningful `alt` text for images
- Ensure sufficient contrast for text overlays
- Consider colorblind-friendly color schemes
- Provide text descriptions for complex diagrams

## Legal Considerations

- Only use images you have permission to use
- Credit sources when required
- Consider using Creative Commons or royalty-free images
- Remove any confidential or sensitive information from screenshots
