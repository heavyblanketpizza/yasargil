// Decoder-only test driver: exercise the actual patched helper without loading
// a language/vision model or using a GPU. Include the implementation so the
// fixture can select its format state directly; production uses projector type.
// Compile with -DMTMD_VIDEO and the patched runtime include/library paths.
#include "mtmd-helper.cpp"

#include <cstring>
#include <iostream>

int main(int argc, char ** argv) {
    if (argc != 4) {
        std::cerr << "usage: qwen-video-stream VIDEO FFMPEG_BIN_DIR qwen|generic|source\n";
        return 2;
    }
    const bool qwen = std::strcmp(argv[3], "qwen") == 0;
    const bool source = std::strcmp(argv[3], "source") == 0;
    if (!qwen && !source && std::strcmp(argv[3], "generic") != 0) {
        return 2;
    }
    mtmd_helper_video ctx;
    ctx.mctx = nullptr;
    ctx.path = argv[1];
    ctx.ffmpeg_bin = video_resolve_bin(argv[2], "ffmpeg");
    ctx.ffprobe_bin = video_resolve_bin(argv[2], "ffprobe");
    ctx.timestamp_interval_ms = 10000;
    if (!ctx.probe(0.0f)) {
        return 3;
    }
    // Independent source comparison omits native's source-FPS filter. Pixel
    // hashes then catch same-count dropping/duplication as well as reordering.
    if (source) {
        ctx.fps_target = 0.0f;
    }
    if (!ctx.start_ffmpeg(0.0f)) {
        return 3;
    }
    ctx.qwen_video_reference = qwen;

    size_t emitted = 0;
    for (;;) {
        mtmd_bitmap * frame = nullptr;
        char * label = nullptr;
        const int result = ctx.read_next(&frame, &label);
        if (result == -1) {
            std::cout << "{\"decoded_frames\":" << ctx.current_frame
                      << ",\"emitted_bitmaps\":" << emitted << "}\n";
            return 0;
        }
        if (result != 0 || (frame != nullptr) == (label != nullptr)) {
            mtmd_bitmap_free(frame);
            free(label);
            return 4;
        }
        if (label) {
            // Both supported formatters contain no JSON escape characters.
            std::cout << "{\"text\":\"" << label << "\"}\n";
            free(label);
        } else {
            mtmd::bitmap_ptr owned(frame);
            const unsigned char * data = mtmd_bitmap_get_data(frame);
            const size_t size = mtmd_bitmap_get_n_bytes(frame);
            std::cout << "{\"bitmap\":" << emitted++ << ",\"rgb_sha256\":\""
                      << hash_sha256_hex(data, size) << "\",\"bytes\":" << size << "}\n";
        }
    }
}
