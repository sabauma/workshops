# [WORKSHOP_TITLE] Workshop

## Overview

[WORKSHOP_DESCRIPTION] - Replace this section with a brief description of your workshop content, learning objectives, and what participants will achieve.

## Prerequisites

### Install Pixi

Follow the [official installation guide](https://pixi.sh/latest/installation/).

## Running the Workshop

### 1. Presentation Slides (Cross-Platform)

The slides work on all supported platforms (Linux, macOS, Windows):

```bash
# Install slide dependencies
pixi run slide-install

# Run development server with hot reload
pixi run dev-slides

# Or run production slides
pixi run slides

# Generate PDF export
pixi run pdf
```

### 2. Interactive Content (Platform-Specific)

[OPTIONAL_SECTION] - Include this section if your workshop has platform-specific components like Jupyter notebooks, GPU examples, or system-specific dependencies.

```bash
# Example for Linux-only content with specific features
pixi shell -e [FEATURE_NAME]

# Start interactive environment
[START_COMMAND]

# Follow the workshop materials and run exercises
[EXERCISE_COMMANDS]

# Exit the pixi environment when done
exit
```

## Supported Platforms

- **Slides**: Linux (x64), macOS (x64/ARM64), Windows (x64)
- **Interactive Content**: [SPECIFY_PLATFORMS] - [DESCRIBE_REQUIREMENTS]

## Workshop Contents

1. **Slides**: [DESCRIBE_SLIDE_CONTENT]
2. **[OPTIONAL_NOTEBOOKS]**: [DESCRIBE_INTERACTIVE_CONTENT]
3. **[OPTIONAL_SCRIPTS]**: [DESCRIBE_SUPPORTING_SCRIPTS]

## System Requirements

[OPTIONAL_SECTION] - Include specific system requirements for your workshop:

### For [FEATURE_A] (e.g., CUDA/NVIDIA):
- Platform requirements
- Software dependencies
- Hardware requirements

### For [FEATURE_B] (e.g., ROCm/AMD):
- Alternative platform support
- Specific versions
- Configuration notes

## Dependencies

The workshop automatically manages dependencies through Pixi, including:
- [LIST_CORE_DEPENDENCIES]
- [LIST_WORKSHOP_SPECIFIC_DEPS]
- [LIST_OPTIONAL_FEATURES]

## Troubleshooting

- **[COMMON_ISSUE_1]**: [SOLUTION_1]
- **[COMMON_ISSUE_2]**: [SOLUTION_2]
- **Pixi environment issues**: Try `pixi clean` and reinstall

## Additional Resources

- **[RELEVANT_DOCS]**: [LINKS_TO_DOCUMENTATION]
- **[COMMUNITY_RESOURCES]**: [FORUM_LINKS]
- **[SOURCE_CODE]**: [REPO_LINKS]

## Contributing

Found an issue or want to improve the workshop content?
1. Fork the repository
2. Create a feature branch
3. Submit a pull request

## License

This project is licensed under the [LICENSE_TYPE] License - see the [LICENSE](../LICENSE) file for details.

---

**Template Usage Notes (Remove this section):**
- Replace all [PLACEHOLDER] text with your workshop-specific content
- Remove optional sections that don't apply to your workshop
- Update the license type and links as appropriate
- Add workshop-specific troubleshooting tips
- Include relevant screenshots or diagrams in the images/ directory
