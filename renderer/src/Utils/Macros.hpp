#pragma once

// Disallow copying
#define DISALLOW_COPY(ClassName) \
    ClassName(const ClassName&) = delete; \
    ClassName& operator=(const ClassName&) = delete;

// Disallow moving
#define DISALLOW_MOVE(ClassName) \
    ClassName(ClassName&&) = delete; \
    ClassName& operator=(ClassName&&) = delete;

    // Allow copying
#define ALLOW_COPY(ClassName) \
    ClassName(const ClassName&) = default; \
    ClassName& operator=(const ClassName&) = default;