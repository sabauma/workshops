# Workshop Plots and Charts Directory

This directory contains generated charts, graphs, and data visualizations for your workshop.

## Recommended Plot Files

### Performance Comparisons:
- `performance-comparison.png` - Before/after performance charts
- `benchmark-results.svg` - Detailed benchmark visualizations
- `scaling-chart.png` - Performance scaling analysis

### Data Visualizations:
- `accuracy-metrics.png` - Model accuracy comparisons
- `resource-usage.svg` - Memory/CPU usage charts
- `timeline-chart.png` - Project timeline or workflow

### Technical Diagrams:
- `system-metrics.png` - System performance monitoring
- `profiling-results.png` - Code profiling visualizations

## File Generation

Most plots should be generated programmatically for consistency:

### Python (Matplotlib, Seaborn, Plotly):
```python
import matplotlib.pyplot as plt
import seaborn as sns

# Generate plot
plt.figure(figsize=(10, 6))
# ... plotting code ...
plt.savefig('plots/performance-comparison.png', dpi=300, bbox_inches='tight')
plt.close()
```

### R (ggplot2):
```r
library(ggplot2)

p <- ggplot(data, aes(x, y)) +
     geom_line() +
     theme_minimal()

ggsave("plots/analysis-results.png", p, width=10, height=6, dpi=300)
```

### JavaScript (D3.js, Chart.js):
```javascript
// Generate interactive charts that can be saved as images
// Consider using headless browser automation for PNG export
```

## Plot Guidelines

### File Formats:
- **PNG**: Standard for most charts (300 DPI for presentations)
- **SVG**: Vector format for simple charts that scale well
- **PDF**: For complex multi-page visualizations
- **WebP**: For web optimization (smaller file sizes)

### Sizing:
- **Presentation plots**: 1200x800px or larger
- **Inline charts**: 800x600px
- **Comparison charts**: 1400x800px for side-by-side
- **High-DPI displays**: 300 DPI minimum

### Design Principles:
- Use colorblind-friendly palettes
- Ensure readability at presentation scale
- Include clear axis labels and legends
- Maintain consistent styling across plots
- Use appropriate chart types for data

### Color Schemes:
```python
# Recommended color palettes
workshop_colors = ['#007acc', '#ff6b35', '#4caf50', '#ff9800', '#9c27b0']
colorblind_friendly = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
```

## Automation

Consider creating a script to regenerate all plots:

```bash
#!/bin/bash
# generate-plots.sh
echo "Generating workshop plots..."
python scripts/generate_performance_plots.py
python scripts/generate_comparison_charts.py
echo "Plots generated successfully!"
```

## Usage in Slides

Reference plots in your presentation:

```markdown
![Performance Comparison](./plots/performance-comparison.png)

<div style="text-align: center;">
<img src="./plots/benchmark-results.svg" alt="Benchmark Results" width="800">
</div>
```

## Data Sources

Document where your plot data comes from:
- Benchmark scripts
- Performance monitoring
- User studies
- Literature references

Keep raw data files in a separate `data/` directory if needed.
