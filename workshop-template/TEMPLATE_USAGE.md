# Workshop Template Usage Guide

This template provides a complete foundation for creating professional workshop presentations based on the successful pytorch-max-bridge and mojo-gpu-programming workshop formats.

## Quick Start

1. **Copy the template:**
   ```bash
   cp -r workshop-template/ my-workshop-name/
   cd my-workshop-name/
   ```

2. **Customize the basics:**
   - Update `package.json` with your workshop details
   - Modify `pixi.toml` for your dependencies
   - Edit `README.md` with your workshop content
   - Replace placeholders in `slides.md`

3. **Test the setup:**
   ```bash
   pixi run slide-install
   pixi run dev-slides
   ```

## Customization Checklist

### Core Files (Required)
- [ ] `package.json` - Update name, description, author, keywords
- [ ] `pixi.toml` - Configure dependencies and environments
- [ ] `README.md` - Replace all [PLACEHOLDER] text
- [ ] `slides.md` - Add your workshop content
- [ ] `index.html` - Update title and hero background image

### Visual Assets
- [ ] Add hero image to `images/` directory (400x400px)
- [ ] Add your photo to `images/` directory (300x300px)
- [ ] Create architecture/solution diagrams
- [ ] Generate performance plots in `plots/`
- [ ] Record demo GIFs for `gifs/` directory

### Advanced Customization
- [ ] Update CSS variables in `index.html` for brand colors
- [ ] Add custom syntax highlighting if needed
- [ ] Configure pixi features for platform-specific content
- [ ] Add workshop-specific dependencies
- [ ] Customize font loading in `fonts/fonts.css`

## Template Structure

```
workshop-template/
├── README.md                  # Workshop documentation
├── TEMPLATE_USAGE.md         # This file (remove when done)
├── package.json              # Node.js dependencies
├── pixi.toml                 # Pixi configuration
├── index.html                # Presentation HTML
├── slides.md                 # Slide content
├── highlight.js              # Custom syntax highlighting
├── mojolang.js              # Mojo language support
├── highlight.css             # Highlighting styles
├── .gitignore                # Git ignore rules
├── .gitattributes            # Git attributes
├── fonts/
│   ├── fonts.css            # Font declarations
│   └── .gitkeep
├── images/
│   ├── README.md            # Image guidelines
│   └── .gitkeep
├── plots/
│   ├── README.md            # Plot generation guide
│   └── .gitkeep
└── gifs/
    ├── README.md            # GIF creation guide
    └── .gitkeep
```

## Placeholder Reference

### Global Placeholders
- `[WORKSHOP_TITLE]` - Full workshop title
- `[WORKSHOP_NAME]` - Short name (lowercase-with-hyphens)
- `[WORKSHOP_DESCRIPTION]` - Brief description
- `[YOUR_NAME]` - Presenter name
- `[YOUR_EMAIL]` - Contact email
- `[COMPANY]` - Organization name

### Content Placeholders
- `[SECTION_N_TITLE]` - Agenda section titles
- `[HERO_IMAGE]` - Main hero image filename
- `[ARCHITECTURE_DIAGRAM]` - System architecture image
- `[SOLUTION_DIAGRAM]` - Solution overview image
- `[DEMO_GIF]` - Demonstration GIF filename

### Technical Placeholders
- `[FEATURE_NAME]` - Pixi environment names
- `[YOUR_LANGUAGE]` - Custom programming language
- `[LICENSE_TYPE]` - License (MIT, Apache, etc.)
- `[REPOSITORY_URL]` - Git repository URL

## Platform Support

The template supports multi-platform development:

### Slides (All Platforms)
- Linux (x64)
- macOS (x64/ARM64)
- Windows (x64)

### Interactive Content (Configurable)
Configure platform-specific features in `pixi.toml`:
- CUDA support (Linux only)
- ROCm support (Linux only)
- Custom hardware features

## Dependencies

### Core Dependencies (All Platforms)
- Node.js 22.13+ (for presentation)
- Reveal.js 5.1+ (slide framework)

### Optional Dependencies
- Python 3.10-3.12 (for notebooks/scripts)
- Jupyter Lab (for interactive content)
- Custom libraries based on workshop needs

## Presentation Features

### Built-in Features
- ✅ Responsive design
- ✅ Code syntax highlighting
- ✅ Speaker notes
- ✅ PDF export
- ✅ Live reload during development
- ✅ Custom CSS styling
- ✅ Multi-language support

### Customizable Features
- Progress indicators
- Interactive demos
- Custom animations
- Brand styling
- Font loading
- Language-specific highlighting

## Development Workflow

### Local Development
```bash
# Install dependencies
pixi run slide-install

# Start development server (auto-reload)
pixi run dev-slides

# Open browser to http://localhost:3034
```

### Production Presentation
```bash
# Start production server
pixi run slides

# Open browser to http://localhost:3033
```

### PDF Export
```bash
# Automated PDF generation
pixi run pdf

# Manual export: Open http://localhost:3033?print-pdf in Chrome
# Use Print → Save as PDF with:
# - Format: Letter or A4
# - Margins: None
# - Background graphics: Enabled
```

## Best Practices

### Content Organization
1. Keep slides concise (6-8 lines max per slide)
2. Use progressive disclosure (fragments)
3. Balance theory with hands-on exercises
4. Include speaker notes for complex topics

### Visual Design
1. Maintain consistent branding
2. Use high-contrast colors
3. Optimize images for web delivery
4. Test readability at presentation scale

### Technical Setup
1. Test on target presentation hardware
2. Have backup plans for demos
3. Ensure reliable internet for resources
4. Practice timing and pacing

### Workshop Delivery
1. Arrive early to test setup
2. Have contact info readily available
3. Encourage questions throughout
4. Provide follow-up resources

## Troubleshooting

### Common Issues

**Slides not loading:**
```bash
# Check port availability
lsof -i :3033 :3034

# Clear npm cache
npm cache clean --force
```

**Font loading issues:**
- Verify font files are in `fonts/` directory
- Check font-face declarations in `fonts/fonts.css`
- Test fallback fonts work properly

**Syntax highlighting problems:**
- Ensure language files are loaded in `index.html`
- Check JavaScript console for errors
- Verify language detection in ready callback

**PDF export issues:**
- Use Chrome browser for best results
- Disable browser extensions that might interfere
- Check for JavaScript errors in console

### Getting Help

- Review original workshop examples
- Check Reveal.js documentation
- Test with minimal content first
- Ask colleagues to review before workshop

## License

This template is provided under the same license as the source workshops. Update the LICENSE file as appropriate for your organization.

---

**Remove this file when your workshop is ready for production use.**
