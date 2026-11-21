// Xbox textures -> host textures. Swizzled (Morton order) and linear layouts are decoded on the CPU into
// A8R8G8B8, DXT data is uploaded as is. Host copies are cached per resource header and re-uploaded when
// the header changes, the game locks the texture, or a sampled hash of the texel data changes (checked at the
// first use in a frame, every 16th frame for textures that have not changed for a while).
#include "d3d/device.h"
#include "d3d/dispatch.h"
#include "d3d/host.h"
#include "d3d/state.h"
#include "perf.h"

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <unordered_set>
#include <cstring>
#include <unordered_map>
#include <vector>

namespace xd3d {
namespace {

struct Entry {
    IDirect3DBaseTexture9 *tex = nullptr;
    uint32_t data = 0, format = 0, size = 0;
    const void *palette = nullptr;
    uint32_t hash = 0;
    bool dirty = false;
    uint32_t checked_frame = ~0u; // the frame the contents were last checked (hashing on every bind cost ~8 ms a frame)
    uint32_t stable = 0;          // consecutive checks without a change
    bool replaced = false;        // the host texture is a replacement from port\\textures (texture packs)
};

// A texture whose contents did not change for this many checks is checked every 16th frame only (busy
// scenes bind hundreds of textures a frame; hashing each once a frame still cost ~3 ms). Textures that
// change (video frames, render-to-memory) stay at one check a frame; a Lock marks a texture dirty at once.
constexpr uint32_t kStableChecks = 30;

std::unordered_map<const Resource *, Entry> g_cache;
CRITICAL_SECTION g_cs;
struct Init {
    Init() { InitializeCriticalSection(&g_cs); }
} g_init;

uint32_t fmt_of(const PixelContainer *pc) { return (pc->Format >> FMT_FORMAT_SHIFT) & 0xFF; }

uint32_t sample_hash(const uint8_t *p, uint32_t bytes)
{
    uint32_t h = 2166136261u;
    uint32_t n = bytes / 4;
    const uint32_t *d = (const uint32_t *)p;
    uint32_t step = n > 8192 ? n / 4096 : 1;
    for (uint32_t i = 0; i < n; i += step) h = (h ^ d[i]) * 16777619u;
    return h;
}

// Morton-order offset tables for one mip level.
struct Swizzle {
    uint32_t mx = 0, my = 0;
    Swizzle(uint32_t w, uint32_t h)
    {
        uint32_t bit = 1;
        for (uint32_t i = 1; i < w || i < h; i <<= 1) {
            if (i < w) { mx |= bit; bit <<= 1; }
            if (i < h) { my |= bit; bit <<= 1; }
        }
    }
    static uint32_t deposit(uint32_t v, uint32_t mask)
    {
        uint32_t r = 0;
        for (uint32_t b = 1; mask; b <<= 1) {
            uint32_t low = mask & (~mask + 1);
            if (v & b) r |= low;
            mask &= mask - 1;
        }
        return r;
    }
};

uint32_t expand5(uint32_t v) { return (v << 3) | (v >> 2); }
uint32_t expand6(uint32_t v) { return (v << 2) | (v >> 4); }
uint32_t expand4(uint32_t v) { return v * 17; }

// Decode one texel of an uncompressed Xbox format into A8R8G8B8.
uint32_t decode(uint32_t fmt, const uint8_t *p, const uint32_t *pal)
{
    uint32_t v16 = p[0] | (p[1] << 8);
    switch (fmt) {
    case XFMT_A8R8G8B8: case XFMT_LIN_A8R8G8B8: return *(const uint32_t *)p;
    case XFMT_X8R8G8B8: case XFMT_LIN_X8R8G8B8: return *(const uint32_t *)p | 0xFF000000;
    case XFMT_R5G6B5: case XFMT_LIN_R5G6B5:
        return 0xFF000000 | expand5(v16 >> 11) << 16 | expand6((v16 >> 5) & 63) << 8 | expand5(v16 & 31);
    case XFMT_A1R5G5B5: case XFMT_LIN_A1R5G5B5:
        return ((v16 & 0x8000) ? 0xFF000000 : 0) | expand5((v16 >> 10) & 31) << 16 | expand5((v16 >> 5) & 31) << 8 | expand5(v16 & 31);
    case XFMT_X1R5G5B5: case XFMT_LIN_X1R5G5B5:
        return 0xFF000000 | expand5((v16 >> 10) & 31) << 16 | expand5((v16 >> 5) & 31) << 8 | expand5(v16 & 31);
    case XFMT_A4R4G4B4: case XFMT_LIN_A4R4G4B4:
        return expand4(v16 >> 12) << 24 | expand4((v16 >> 8) & 15) << 16 | expand4((v16 >> 4) & 15) << 8 | expand4(v16 & 15);
    case XFMT_R6G5B5: case XFMT_LIN_R6G5B5:
        return 0xFF000000 | expand6(v16 >> 10) << 16 | expand5((v16 >> 5) & 31) << 8 | expand5(v16 & 31);
    case XFMT_L8: case XFMT_LIN_L8: return 0xFF000000 | p[0] * 0x010101u;
    case XFMT_AL8: case XFMT_LIN_AL8: return p[0] * 0x01010101u;
    case XFMT_A8: case XFMT_LIN_A8: return (uint32_t)p[0] << 24 | 0xFFFFFF;
    case XFMT_A8L8: case XFMT_LIN_A8L8: return (uint32_t)p[1] << 24 | p[0] * 0x010101u;
    case XFMT_L16: case XFMT_LIN_L16: return 0xFF000000 | p[1] * 0x010101u;
    case XFMT_P8: return pal ? pal[p[0]] : 0xFF000000 | p[0] * 0x010101u;
    case XFMT_R8B8: case XFMT_LIN_R8B8: return 0xFF000000 | (uint32_t)p[1] << 16 | p[0];
    case XFMT_G8B8: case XFMT_LIN_G8B8: return 0xFF000000 | (uint32_t)p[1] << 8 | p[0];
    case XFMT_V8U8: return 0xFF000000 | ((p[0] ^ 0x80u) << 16) | ((p[1] ^ 0x80u) << 8);
    case XFMT_A8B8G8R8: case XFMT_LIN_A8B8G8R8: { uint32_t v = *(const uint32_t *)p; return (v & 0xFF00FF00) | (v & 0xFF) << 16 | (v >> 16 & 0xFF); }
    case XFMT_B8G8R8A8: case XFMT_LIN_B8G8R8A8: { uint32_t v = *(const uint32_t *)p; return (v << 24) | (v >> 8); }
    case XFMT_R8G8B8A8: case XFMT_LIN_R8G8B8A8: { uint32_t v = *(const uint32_t *)p; return (v << 24) | (v >> 24 & 0xFF) << 16 | (v >> 16 & 0xFF) << 8 | (v >> 8 & 0xFF); }
    case XFMT_D24S8: case XFMT_LIN_D24S8: case XFMT_F24S8: case XFMT_LIN_F24S8: { uint32_t d = *(const uint32_t *)p >> 24; return 0xFF000000 | d * 0x010101u; }
    case XFMT_D16: case XFMT_LIN_D16: case XFMT_F16: case XFMT_LIN_F16: return 0xFF000000 | p[1] * 0x010101u;
    default: return 0xFFFF00FF; // magenta: format not handled
    }
}

uint32_t yuv_to_rgb(int y, int u, int v)
{
    int c = y - 16, d = u - 128, e = v - 128;
    auto clamp = [](int x) { return x < 0 ? 0 : x > 255 ? 255 : x; };
    int r = clamp((298 * c + 409 * e + 128) >> 8), g = clamp((298 * c - 100 * d - 208 * e + 128) >> 8),
        b = clamp((298 * c + 516 * d + 128) >> 8);
    return 0xFF000000 | r << 16 | g << 8 | b;
}

// Write level `level` of `pc` (from Xbox memory) into a locked host rect.
void upload_level(const PixelContainer *pc, uint32_t face, uint32_t level, uint8_t *dst, uint32_t dst_pitch, const uint32_t *pal)
{
    uint32_t fmt = fmt_of(pc);
    const FormatInfo &fi = format_info(fmt);
    uint32_t pitch, w, h;
    uint32_t off = texture_level_offset(pc, face, level, &pitch, &w, &h);
    const uint8_t *src = (const uint8_t *)phys_to_virt(pc->Data) + off;
    if (fi.compressed) {
        uint32_t rows = (h + 3) / 4;
        for (uint32_t y = 0; y < rows; y++) memcpy(dst + y * dst_pitch, src + y * pitch, pitch < dst_pitch ? pitch : dst_pitch);
        return;
    }
    uint32_t bpp = fi.bpp / 8;
    if (fmt == XFMT_YUY2 || fmt == XFMT_UYVY) {
        for (uint32_t y = 0; y < h; y++) {
            const uint8_t *s = src + y * pitch;
            uint32_t *d = (uint32_t *)(dst + y * dst_pitch);
            for (uint32_t x = 0; x + 1 < w; x += 2, s += 4) {
                int y0, y1, u, v;
                if (fmt == XFMT_YUY2) { y0 = s[0]; u = s[1]; y1 = s[2]; v = s[3]; }
                else { u = s[0]; y0 = s[1]; v = s[2]; y1 = s[3]; }
                d[x] = yuv_to_rgb(y0, u, v);
                d[x + 1] = yuv_to_rgb(y1, u, v);
            }
        }
        return;
    }
    if (fi.linear) {
        for (uint32_t y = 0; y < h; y++) {
            const uint8_t *s = src + y * pitch;
            uint32_t *d = (uint32_t *)(dst + y * dst_pitch);
            for (uint32_t x = 0; x < w; x++) d[x] = decode(fmt, s + x * bpp, pal);
        }
        return;
    }
    Swizzle sw(w, h);
    for (uint32_t y = 0; y < h; y++) {
        uint32_t oy = Swizzle::deposit(y, sw.my);
        uint32_t *d = (uint32_t *)(dst + y * dst_pitch);
        uint32_t ox = 0;
        for (uint32_t x = 0; x < w; x++) {
            d[x] = decode(fmt, src + (ox | oy) * bpp, pal);
            ox = (ox - sw.mx) & sw.mx; // next x in Morton order
        }
    }
}

D3DFORMAT host_format(uint32_t fmt)
{
    switch (fmt) {
    case XFMT_DXT1: return D3DFMT_DXT1;
    case XFMT_DXT3: return D3DFMT_DXT3;
    case XFMT_DXT5: return D3DFMT_DXT5;
    default: return D3DFMT_A8R8G8B8;
    }
}

uint32_t total_bytes(const PixelContainer *pc)
{
    uint32_t w, h, d, levels;
    pixel_container_dims(pc, &w, &h, &d, &levels);
    uint32_t pitch, lw, lh;
    bool cube = (pc->Format & FMT_CUBEMAP) != 0;
    uint32_t last = texture_level_offset(pc, cube ? 5 : 0, levels - 1, &pitch, &lw, &lh);
    const FormatInfo &fi = format_info(fmt_of(pc));
    return last + pitch * (fi.compressed ? (lh + 3) / 4 : lh);
}

// Refill an existing host texture (same size/format) from Xbox memory.
void refill(IDirect3DBaseTexture9 *base, Resource *r, Palette *pal)
{
    auto *pc = (PixelContainer *)r;
    const uint32_t *palette = pal ? (const uint32_t *)phys_to_virt(pal->Data) : nullptr;
    if (base->GetType() == D3DRTYPE_CUBETEXTURE) {
        auto *t = (IDirect3DCubeTexture9 *)base;
        for (uint32_t f = 0; f < 6; f++)
            for (uint32_t l = 0; l < t->GetLevelCount(); l++) {
                D3DLOCKED_RECT lr;
                if (SUCCEEDED(t->LockRect((D3DCUBEMAP_FACES)f, l, &lr, nullptr, 0))) {
                    upload_level(pc, f, l, (uint8_t *)lr.pBits, lr.Pitch, palette);
                    t->UnlockRect((D3DCUBEMAP_FACES)f, l);
                }
            }
        return;
    }
    auto *t = (IDirect3DTexture9 *)base;
    for (uint32_t l = 0; l < t->GetLevelCount(); l++) {
        D3DLOCKED_RECT lr;
        if (SUCCEEDED(t->LockRect(l, &lr, nullptr, 0))) {
            upload_level(pc, 0, l, (uint8_t *)lr.pBits, lr.Pitch, palette);
            t->UnlockRect(l);
        }
    }
}


// ---- Texture packs ---------------------------------------------------------------------------------------------
// A 2D swizzled texture is named by a 64-bit FNV-1a hash of its top level's bytes, its format and size:
// port\textures_dump\<hash>.dds receives the game's texture (texdump = 1; DXT stays DXT, other formats are
// decoded to A8R8G8B8) and port\textures\<hash>.dds, when present, is loaded instead (any size, A8R8G8B8 or
// DXT1/3/5, with its own mip chain). Linear textures (render targets, video, screen-space data) are left alone.

#pragma pack(push, 1)
struct DdsPixelFormat { uint32_t size, flags, fourcc, bits, r, g, b, a; };
struct DdsHeader {
    uint32_t size, flags, height, width, pitch, depth, mips, reserved1[11];
    DdsPixelFormat pf;
    uint32_t caps, caps2, caps3, caps4, reserved2;
};
#pragma pack(pop)
constexpr uint32_t kFourCC_DXT1 = 0x31545844, kFourCC_DXT3 = 0x33545844, kFourCC_DXT5 = 0x35545844;

std::unordered_set<uint64_t> g_pack;     // hashes with a file in port\textures
std::unordered_set<uint64_t> g_dumped;   // hashes written (or found) in port\textures_dump this run
bool g_pack_scanned = false;
std::atomic<int> g_pack_loaded{0}, g_pack_dumps{0};

std::wstring pack_dir() { return g_cfg.port_dir + L"\\textures"; }
std::wstring dump_dir() { return g_cfg.port_dir + L"\\textures_dump"; }

std::wstring hash_name(uint64_t h)
{
    wchar_t b[24];
    swprintf_s(b, L"%016llx.dds", (unsigned long long)h);
    return b;
}

void scan_dir(const std::wstring &dir, std::unordered_set<uint64_t> &out)
{
    WIN32_FIND_DATAW fd;
    HANDLE f = FindFirstFileW((dir + L"\\*.dds").c_str(), &fd);
    if (f == INVALID_HANDLE_VALUE) return;
    do {
        if (wcslen(fd.cFileName) == 20) out.insert(wcstoull(fd.cFileName, nullptr, 16));
    } while (FindNextFileW(f, &fd));
    FindClose(f);
}

void pack_scan()
{
    if (g_pack_scanned) return;
    g_pack_scanned = true;
    if (g_cfg.tex_replace) scan_dir(pack_dir(), g_pack);
    if (g_cfg.tex_dump) {
        CreateDirectoryW(dump_dir().c_str(), nullptr);
        scan_dir(dump_dir(), g_dumped);
    }
    logf("[tex] texture pack: %u replacement(s) in textures\\%s, anisotropic filtering %dx", (unsigned)g_pack.size(),
         g_cfg.tex_dump ? ", dumping new textures to textures_dump\\" : "", g_cfg.aniso);
}

uint64_t content_hash(const PixelContainer *pc, uint32_t fmt, uint32_t w, uint32_t h)
{
    uint32_t pitch, lw, lh;
    texture_level_offset(pc, 0, 0, &pitch, &lw, &lh);
    const FormatInfo &fi = format_info(fmt);
    uint32_t bytes = pitch * (fi.compressed ? (lh + 3) / 4 : lh);
    const uint8_t *p = (const uint8_t *)phys_to_virt(pc->Data);
    uint64_t x = 14695981039346656037ull;
    auto mix = [&](uint32_t v) { for (int i = 0; i < 4; i++) { x ^= (v >> (i * 8)) & 0xFF; x *= 1099511628211ull; } };
    mix(fmt); mix(w); mix(h);
    for (uint32_t i = 0; i < bytes; i++) { x ^= p[i]; x *= 1099511628211ull; }
    return x;
}

void dump_texture(const PixelContainer *pc, uint32_t fmt, uint32_t w, uint32_t h, uint64_t hash, const uint32_t *pal)
{
    if (g_dumped.count(hash)) return;
    g_dumped.insert(hash);
    const FormatInfo &fi = format_info(fmt);
    DdsHeader hd = {};
    hd.size = 124;
    hd.flags = 0x1 | 0x2 | 0x4 | 0x1000; // caps, height, width, pixelformat
    hd.width = w;
    hd.height = h;
    hd.mips = 1;
    hd.pf.size = 32;
    hd.caps = 0x1000;
    std::vector<uint8_t> data;
    if (fi.compressed) {
        uint32_t pitch, lw, lh;
        uint32_t off = texture_level_offset(pc, 0, 0, &pitch, &lw, &lh);
        uint32_t bytes = pitch * ((lh + 3) / 4);
        const uint8_t *src = (const uint8_t *)phys_to_virt(pc->Data) + off;
        data.assign(src, src + bytes);
        hd.flags |= 0x80000; // linearsize
        hd.pitch = bytes;
        hd.pf.flags = 0x4;   // fourcc
        hd.pf.fourcc = fmt == XFMT_DXT1 ? kFourCC_DXT1 : fmt == XFMT_DXT3 ? kFourCC_DXT3 : kFourCC_DXT5;
    } else {
        data.resize((size_t)w * h * 4);
        upload_level(pc, 0, 0, data.data(), w * 4, pal);
        hd.flags |= 0x8; // pitch
        hd.pitch = w * 4;
        hd.pf.flags = 0x41; // rgb + alpha
        hd.pf.bits = 32;
        hd.pf.r = 0x00FF0000; hd.pf.g = 0x0000FF00; hd.pf.b = 0x000000FF; hd.pf.a = 0xFF000000;
    }
    std::wstring path = dump_dir() + L"\\" + hash_name(hash);
    FILE *f = nullptr;
    if (_wfopen_s(&f, path.c_str(), L"wb") || !f) return;
    uint32_t magic = 0x20534444;
    fwrite(&magic, 4, 1, f);
    fwrite(&hd, sizeof hd, 1, f);
    fwrite(data.data(), 1, data.size(), f);
    fclose(f);
    if (++g_pack_dumps % 100 == 1) logf("[tex] dumped %d texture(s) so far", g_pack_dumps.load());
}

IDirect3DTexture9 *load_replacement(uint64_t hash)
{
    std::wstring path = pack_dir() + L"\\" + hash_name(hash);
    FILE *f = nullptr;
    if (_wfopen_s(&f, path.c_str(), L"rb") || !f) return nullptr;
    std::vector<uint8_t> file;
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (n > 128) { file.resize(n); fread(file.data(), 1, n, f); }
    fclose(f);
    if (file.size() < 128 || *(const uint32_t *)file.data() != 0x20534444) return nullptr;
    const DdsHeader &hd = *(const DdsHeader *)(file.data() + 4);
    D3DFORMAT hf;
    uint32_t block = 0; // bytes per 4x4 block, 0 = 32-bit uncompressed
    if (hd.pf.flags & 0x4) {
        if (hd.pf.fourcc == kFourCC_DXT1) { hf = D3DFMT_DXT1; block = 8; }
        else if (hd.pf.fourcc == kFourCC_DXT3) { hf = D3DFMT_DXT3; block = 16; }
        else if (hd.pf.fourcc == kFourCC_DXT5) { hf = D3DFMT_DXT5; block = 16; }
        else return nullptr;
    } else if (hd.pf.bits == 32 && hd.pf.r == 0x00FF0000 && hd.pf.b == 0x000000FF) hf = D3DFMT_A8R8G8B8;
    else return nullptr;
    uint32_t w = hd.width, h = hd.height, mips = (hd.flags & 0x20000) && hd.mips ? hd.mips : 1;
    IDirect3DTexture9 *t = nullptr;
    if (FAILED(host3d::dev()->CreateTexture(w, h, mips, 0, hf, D3DPOOL_MANAGED, &t, nullptr))) return nullptr;
    size_t off = 128;
    for (uint32_t l = 0; l < mips; l++) {
        uint32_t lw = std::max(1u, w >> l), lh = std::max(1u, h >> l);
        uint32_t rows = block ? (lh + 3) / 4 : lh, row = block ? ((lw + 3) / 4) * block : lw * 4;
        if (off + (size_t)rows * row > file.size()) { t->Release(); return nullptr; }
        D3DLOCKED_RECT lr;
        if (SUCCEEDED(t->LockRect(l, &lr, nullptr, 0))) {
            for (uint32_t y = 0; y < rows; y++) memcpy((uint8_t *)lr.pBits + y * lr.Pitch, file.data() + off + (size_t)y * row, row);
            t->UnlockRect(l);
        }
        off += (size_t)rows * row;
    }
    if (++g_pack_loaded <= 20 || g_pack_loaded % 200 == 0)
        logf("[tex] replacement %016llx: %ux%u x%u (%d loaded)", (unsigned long long)hash, w, h, mips, g_pack_loaded.load());
    return t;
}

IDirect3DBaseTexture9 *create_and_upload(Resource *r, Palette *pal, bool *replaced)
{
    *replaced = false;
    auto *pc = (PixelContainer *)r;
    uint32_t fmt = fmt_of(pc);
    uint32_t w, h, d, levels;
    pixel_container_dims(pc, &w, &h, &d, &levels);
    if (levels == 0) levels = 1;
    bool cube = (pc->Format & FMT_CUBEMAP) != 0;
    uint32_t dim = (pc->Format >> FMT_DIMENSION_SHIFT) & 0xF;
    if (dim == 3) {
        KNOTE_ONCE("volume textures are not uploaded yet (%ux%ux%u fmt %s)", w, h, d, format_info(fmt).name);
        return nullptr;
    }
    const uint32_t *palette = pal ? (const uint32_t *)phys_to_virt(pal->Data) : nullptr;
    D3DFORMAT hf = host_format(fmt);
    IDirect3DDevice9 *dev = host3d::dev();
    if (cube) {
        IDirect3DCubeTexture9 *t = nullptr;
        if (FAILED(dev->CreateCubeTexture(w, levels, 0, hf, D3DPOOL_MANAGED, &t, nullptr))) return nullptr;
        for (uint32_t f = 0; f < 6; f++)
            for (uint32_t l = 0; l < levels; l++) {
                D3DLOCKED_RECT lr;
                if (SUCCEEDED(t->LockRect((D3DCUBEMAP_FACES)f, l, &lr, nullptr, 0))) {
                    upload_level(pc, f, l, (uint8_t *)lr.pBits, lr.Pitch, palette);
                    t->UnlockRect((D3DCUBEMAP_FACES)f, l);
                }
            }
        return t;
    }
    if ((g_cfg.tex_replace || g_cfg.tex_dump) && !pc->Size && fmt != XFMT_YUY2 && fmt != XFMT_UYVY && w >= 4 && h >= 4) {
        pack_scan();
        bool want = g_cfg.tex_replace && !g_pack.empty();
        if (want || g_cfg.tex_dump) {
            uint64_t ch = content_hash(pc, fmt, w, h);
            if (want && g_pack.count(ch)) {
                if (IDirect3DTexture9 *rt = load_replacement(ch)) {
                    *replaced = true;
                    return rt;
                }
            }
            if (g_cfg.tex_dump) dump_texture(pc, fmt, w, h, ch, palette);
        }
    }
    IDirect3DTexture9 *t = nullptr;
    if (FAILED(dev->CreateTexture(w, h, levels, 0, hf, D3DPOOL_MANAGED, &t, nullptr))) {
        logf("[tex] CreateTexture %ux%u x%u %s failed", w, h, levels, format_info(fmt).name);
        return nullptr;
    }
    for (uint32_t l = 0; l < levels; l++) {
        D3DLOCKED_RECT lr;
        if (SUCCEEDED(t->LockRect(l, &lr, nullptr, 0))) {
            upload_level(pc, 0, l, (uint8_t *)lr.pBits, lr.Pitch, palette);
            t->UnlockRect(l);
        }
    }
    return t;
}

} // namespace

void pixel_container_dims(const PixelContainer *pc, uint32_t *w, uint32_t *h, uint32_t *d, uint32_t *levels)
{
    if (pc->Size) {
        *w = (pc->Size & SIZE_WIDTH_MASK) + 1;
        *h = ((pc->Size >> SIZE_HEIGHT_SHIFT) & 0xFFF) + 1;
        *d = 1;
        *levels = 1;
    } else {
        *w = 1u << ((pc->Format >> FMT_USIZE_SHIFT) & 0xF);
        *h = 1u << ((pc->Format >> FMT_VSIZE_SHIFT) & 0xF);
        *d = 1u << ((pc->Format >> FMT_PSIZE_SHIFT) & 0xF);
        *levels = (pc->Format >> FMT_MIPMAP_SHIFT) & 0xF;
        if (!*levels) *levels = 1;
    }
}

uint32_t texture_level_offset(const PixelContainer *pc, uint32_t face, uint32_t level, uint32_t *pitch, uint32_t *w, uint32_t *h)
{
    uint32_t tw, th, td, levels;
    pixel_container_dims(pc, &tw, &th, &td, &levels);
    const FormatInfo &fi = format_info(fmt_of(pc));
    if (pc->Size) { // linear: one level
        *pitch = (((pc->Size >> SIZE_PITCH_SHIFT) & 0xFF) + 1) * 64;
        *w = tw;
        *h = th;
        return 0;
    }
    auto level_bytes = [&](uint32_t l, uint32_t *p, uint32_t *lw, uint32_t *lh) {
        uint32_t x = tw >> l, y = th >> l;
        if (!x) x = 1;
        if (!y) y = 1;
        *lw = x;
        *lh = y;
        if (fi.compressed) {
            uint32_t bw = (x + 3) / 4, bh = (y + 3) / 4;
            *p = bw * fi.bpp * 2;
            return *p * bh;
        }
        *p = x * fi.bpp / 8;
        return *p * y;
    };
    uint32_t off = 0, p, lw, lh;
    if (face) {
        uint32_t face_bytes = 0;
        for (uint32_t l = 0; l < levels; l++) face_bytes += level_bytes(l, &p, &lw, &lh);
        face_bytes = (face_bytes + 127) & ~127u; // cube faces are 128-byte aligned
        off = face * face_bytes;
    }
    for (uint32_t l = 0; l < level; l++) off += level_bytes(l, &p, &lw, &lh);
    level_bytes(level, pitch, w, h);
    return off;
}

IDirect3DBaseTexture9 *host_texture(Resource *r, Palette *pal)
{
    if (!r || !r->Data) return nullptr;
    perf::Scope perf_scope(perf::kTexture);
    auto *pc = (PixelContainer *)r;
    EnterCriticalSection(&g_cs);
    Entry &e = g_cache[r];
    uint32_t frame = g_frame_count.load(std::memory_order_relaxed);
    bool skip = e.checked_frame == frame || (e.stable >= kStableChecks && ((frame + ((uintptr_t)r >> 6)) & 15) != 0);
    if (skip && e.tex && !e.dirty && e.data == pc->Data && e.format == pc->Format && e.size == pc->Size &&
        e.palette == (pal ? phys_to_virt(pal->Data) : nullptr)) {
        IDirect3DBaseTexture9 *t = e.tex;
        LeaveCriticalSection(&g_cs);
        return t;
    }
    e.checked_frame = frame;
    uint32_t bytes = total_bytes(pc);
    uint32_t hash = sample_hash((const uint8_t *)phys_to_virt(pc->Data), bytes);
    const void *palptr = pal ? phys_to_virt(pal->Data) : nullptr;
    if (fmt_of(pc) == XFMT_P8 && pal) hash ^= sample_hash((const uint8_t *)palptr, 1024);
    bool same_shape = e.tex && e.data == pc->Data && e.format == pc->Format && e.size == pc->Size;
    if (e.replaced && (e.dirty || e.hash != hash)) same_shape = false; // a replacement cannot be refilled in place
    bool stale = !same_shape || e.dirty || e.hash != hash || (fmt_of(pc) == XFMT_P8 && e.palette != palptr);
    e.stable = stale ? 0 : std::min<uint32_t>(e.stable + 1, kStableChecks);
    if (stale && same_shape) { // contents changed (video frames, render-to-memory): refill in place
        refill(e.tex, r, pal);
        perf::add_texture_upload(bytes);
        e.palette = palptr;
        e.hash = hash;
        e.dirty = false;
        stale = false;
    }
    if (stale) {
        if (e.tex) e.tex->Release();
        e.tex = create_and_upload(r, pal, &e.replaced);
        perf::add_texture_upload(bytes);
        e.data = pc->Data;
        e.format = pc->Format;
        e.size = pc->Size;
        e.palette = palptr;
        e.hash = hash;
        e.dirty = false;
        static std::atomic<int> uploads{0};
        if (++uploads <= 40 || g_trace_d3d) {
            uint32_t w, h, d, l;
            pixel_container_dims(pc, &w, &h, &d, &l);
            logf("[tex] upload %p: %ux%u x%u %s%s, data 0x%08x", r, w, h, l, format_info(fmt_of(pc)).name,
                 (pc->Format & FMT_CUBEMAP) ? " cube" : "", pc->Data);
        }
    }
    IDirect3DBaseTexture9 *t = e.tex;
    LeaveCriticalSection(&g_cs);
    return t;
}

void texture_mark_dirty(const Resource *r)
{
    EnterCriticalSection(&g_cs);
    auto it = g_cache.find(r);
    if (it != g_cache.end()) it->second.dirty = true;
    LeaveCriticalSection(&g_cs);
}

void texture_forget(const Resource *r)
{
    EnterCriticalSection(&g_cs);
    auto it = g_cache.find(r);
    if (it != g_cache.end()) {
        if (it->second.tex) it->second.tex->Release();
        g_cache.erase(it);
    }
    LeaveCriticalSection(&g_cs);
}

} // namespace xd3d
