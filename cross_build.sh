#!/bin/bash
# ============================================================
# Ubuntu → Windows 10 EXE Cross-Compile Script (v3)
# ============================================================
# Python 3.11 Embeddable ZIP + Wine
# - 3.11'de _wmi.pyd yok → Wine 6.x'te sorunsuz
# - cvxpy cp311-win_amd64 wheel mevcut → derleme gereksiz
#
#   ./cross_build.sh setup    # Bir kere: Wine + Win Python kur
#   ./cross_build.sh install  # Bir kere: Kütüphaneleri kur
#   ./cross_build.sh build    # Her seferinde: EXE üret
#   ./cross_build.sh test     # Kurulumu kontrol et
#   ./cross_build.sh clean    # Build artıklarını sil
# ============================================================

set -e
export WINEDEBUG=-all
export WINEPREFIX="$HOME/.wine"

# =================== CONFIG - DÜZENLE ===================

MAIN_SCRIPT="main.py"          # Ana dosyan
EXE_NAME="MyApp"               # Çıkacak exe adı
ONE_FILE=true                   # true: tek .exe / false: klasör
SHOW_CONSOLE=false              # true: cmd açılır / false: sadece GUI

# Ek data dosyaları (opsiyonel)
EXTRA_DATA=()
# Örnek: EXTRA_DATA=("assets;assets" "config.json;.")

# =================== PYTHON AYARLARI ====================

# Python 3.11.8 - Wine 6.x uyumlu, cvxpy wheel mevcut
# 3.11.9 embed zip'i 404 verirse 3.11.8'i dene
PY_VER="3.11.8"
PY_VER_SHORT="311"
PY_ZIP="python-${PY_VER}-embed-amd64.zip"
PY_URL="https://www.python.org/ftp/python/${PY_VER}/${PY_ZIP}"
GETPIP_URL="https://bootstrap.pypa.io/get-pip.py"

# Wine içindeki Python dizini
PY_DIR="$WINEPREFIX/drive_c/python${PY_VER_SHORT}"
WINPY="$PY_DIR/python.exe"

G='\033[0;32m'; Y='\033[1;33m'; R='\033[0;31m'; N='\033[0m'
info()  { echo -e "${G}[*]${N} $1"; }
warn()  { echo -e "${Y}[!]${N} $1"; }
die()   { echo -e "${R}[HATA]${N} $1"; exit 1; }

# =================== SETUP ===============================

do_setup() {
    info "=== Wine + Windows Python ${PY_VER} Kurulumu ==="

    # 1) Wine
    if ! command -v wine &>/dev/null; then
        info "Wine kuruluyor..."
        sudo dpkg --add-architecture i386
        sudo apt update
        sudo apt install -y wine64 wine32
    fi
    info "Wine: $(wine64 --version 2>/dev/null || wine --version 2>/dev/null)"

    wineboot --init 2>/dev/null || true
    sleep 2

    # 2) Python embeddable zip
    if [ -f "$WINPY" ]; then
        info "Windows Python zaten kurulu: $PY_DIR"
    else
        info "Python ${PY_VER} embeddable zip indiriliyor..."
        wget -q --show-progress -O "/tmp/${PY_ZIP}" "$PY_URL"
        if [ $? -ne 0 ]; then
            # 3.11.8 yoksa 3.11.7'yi dene
            warn "${PY_VER} bulunamadı, 3.11.7 deneniyor..."
            PY_VER="3.11.7"
            PY_ZIP="python-${PY_VER}-embed-amd64.zip"
            PY_URL="https://www.python.org/ftp/python/${PY_VER}/${PY_ZIP}"
            wget -q --show-progress -O "/tmp/${PY_ZIP}" "$PY_URL" || \
                die "İndirme başarısız!"
        fi

        info "Extract ediliyor..."
        mkdir -p "$PY_DIR"
        unzip -o -q "/tmp/${PY_ZIP}" -d "$PY_DIR"
        rm -f "/tmp/${PY_ZIP}"

        # 3) _pth dosyasını düzenle
        PTH_FILE="$PY_DIR/python${PY_VER_SHORT}._pth"
        if [ -f "$PTH_FILE" ]; then
            info "_pth dosyası düzenleniyor..."
            sed -i 's/^#import site/import site/' "$PTH_FILE"
            if ! grep -q "Lib/site-packages" "$PTH_FILE"; then
                echo "Lib/site-packages" >> "$PTH_FILE"
            fi
        else
            die "_pth dosyası bulunamadı: $PTH_FILE"
        fi

        mkdir -p "$PY_DIR/Lib/site-packages"

        # 4) pip kur
        info "pip kuruluyor..."
        wget -q --show-progress -O "/tmp/get-pip.py" "$GETPIP_URL"
        wine "$WINPY" "/tmp/get-pip.py" 2>/dev/null
        if [ $? -ne 0 ]; then
            die "pip kurulumu başarısız! Hata detayı için:\n  export WINEDEBUG=err+all\n  wine $WINPY /tmp/get-pip.py"
        fi
        rm -f "/tmp/get-pip.py"
    fi

    # Kontrol
    echo ""
    info "Kontrol:"
    wine "$WINPY" --version 2>/dev/null
    wine "$WINPY" -m pip --version 2>/dev/null
    echo ""
    info "Başarılı! Şimdi './cross_build.sh install' çalıştır."
}

# =================== INSTALL =============================

do_install() {
    [ -f "$WINPY" ] || die "Windows Python bulunamadı. Önce 'setup' çalıştır."

    info "=== Kütüphaneler Kuruluyor ==="
    info "Tümü wheel olarak inecek, derleme yok."
    echo ""

    wine "$WINPY" -m pip install --upgrade pip 2>/dev/null

    # --only-binary :all: ile sadece hazır wheel'leri kullan
    # Derleme denemesi olmasın (Wine'da compiler yok)
    PKGS=(pyinstaller numpy scipy matplotlib cvxpy pygame clarabel pillow)

    for pkg in "${PKGS[@]}"; do
        info "Kuruluyor: $pkg"
        wine "$WINPY" -m pip install --only-binary :all: "$pkg" 2>/dev/null && \
            echo -e "  ${G}OK${N}" || {
            warn "  $pkg wheel bulunamadı, source ile deneniyor..."
            wine "$WINPY" -m pip install "$pkg" 2>/dev/null && \
                echo -e "  ${G}OK${N}" || \
                warn "  $pkg KURULAMADI"
        }
    done

    echo ""
    info "Kurulum tamam. './cross_build.sh test' ile kontrol et."
}

# =================== BUILD ===============================

do_build() {
    [ -f "$WINPY" ] || die "Windows Python bulunamadı. Önce 'setup' çalıştır."
    [ -f "$MAIN_SCRIPT" ] || die "'$MAIN_SCRIPT' bulunamadı! CONFIG bölümünü düzenle."

    info "=== Build: $MAIN_SCRIPT → $EXE_NAME.exe ==="

    ARGS=(
        -m PyInstaller
        --name "$EXE_NAME"
        --noconfirm --clean
    )

    $ONE_FILE && ARGS+=(--onefile) || ARGS+=(--onedir)
    $SHOW_CONSOLE || ARGS+=(--noconsole)

    # Hidden imports
    local HI=(
        # numpy
        numpy numpy.core._methods numpy.core._dtype_ctypes numpy.random
        # scipy
        scipy scipy.sparse scipy.sparse.linalg scipy.optimize
        scipy.linalg scipy.interpolate scipy.special scipy.integrate
        # matplotlib
        matplotlib matplotlib.pyplot
        matplotlib.backends.backend_tkagg matplotlib.backends.backend_agg
        # cvxpy + solvers
        cvxpy cvxpy.atoms cvxpy.constraints cvxpy.problems
        cvxpy.reductions cvxpy.cvxcore cvxpy.cvxcore.python
        scs ecos osqp clarabel
        # pygame
        pygame pygame.display pygame.event pygame.font
        pygame.image pygame.mixer pygame.draw pygame.transform
        # misc
        PIL pkg_resources packaging packaging.version
    )
    for h in "${HI[@]}"; do ARGS+=(--hidden-import="$h"); done

    # Collect data/submodules
    ARGS+=(
        --collect-data matplotlib
        --collect-submodules matplotlib
        --collect-all cvxpy
        --collect-all scs
        --collect-all ecos
        --collect-data pygame
    )

    # Clarabel varsa ekle
    wine "$WINPY" -c "import clarabel" 2>/dev/null && ARGS+=(--collect-all clarabel)

    # Ek data dosyaları
    for d in "${EXTRA_DATA[@]}"; do ARGS+=(--add-data="$d"); done

    ARGS+=("$MAIN_SCRIPT")

    info "PyInstaller çalışıyor..."
    echo ""
    wine "$WINPY" "${ARGS[@]}" 2>&1 | \
        grep -E "(INFO|WARN|ERROR|Build|EXE|PKG|Traceback|Error)" || true

    # Sonuç kontrol
    local EXE
    $ONE_FILE && EXE="dist/${EXE_NAME}.exe" || EXE="dist/${EXE_NAME}/${EXE_NAME}.exe"

    if [ -f "$EXE" ]; then
        local SIZE=$(du -h "$EXE" | cut -f1)
        echo ""
        echo "══════════════════════════════════════════"
        echo -e "  ${G}BUILD BAŞARILI!${N}"
        echo "  EXE:   $(realpath "$EXE")"
        echo "  Boyut: $SIZE"
        echo "  Hedef: Windows 10 x64"
        echo "══════════════════════════════════════════"
    else
        die "EXE oluşturulamadı. Logları kontrol et."
    fi
}

# =================== TEST ================================

do_test() {
    [ -f "$WINPY" ] || die "Windows Python bulunamadı."

    info "=== Kurulum Testi ==="
    wine "$WINPY" -c "
import sys
print(f'Python {sys.version}')
print(f'Platform: {sys.platform}')
print()

checks = {
    'numpy': 'numpy',
    'scipy': 'scipy',
    'matplotlib': 'matplotlib',
    'cvxpy': 'cvxpy',
    'pygame': 'pygame',
    'clarabel': 'clarabel',
    'PyInstaller': 'PyInstaller',
}

ok = 0
for name, mod in checks.items():
    try:
        m = __import__(mod)
        v = getattr(m, '__version__', '?')
        print(f'  {name:15s} {v:10s} OK')
        ok += 1
    except ImportError as e:
        print(f'  {name:15s} EKSIK ({e})')

print(f'\n{ok}/{len(checks)} modul hazir')
" 2>/dev/null
}

# =================== CLEAN / NUKE ========================

do_clean() {
    info "Build artıkları temizleniyor..."
    rm -rf build/ dist/ __pycache__/ *.spec
    info "Temiz."
}

do_nuke() {
    warn "Wine Python tamamen kaldırılıyor: $PY_DIR"
    rm -rf "$PY_DIR"
    info "Kaldırıldı. 'setup' ile tekrar kurabilirsin."
}

# =================== MAIN ================================

case "${1:-}" in
    setup)   do_setup   ;;
    install) do_install  ;;
    build)   do_build    ;;
    test)    do_test     ;;
    clean)   do_clean    ;;
    nuke)    do_nuke     ;;
    *)
        echo "Ubuntu → Windows 10 EXE Cross-Compiler (v3)"
        echo "Python ${PY_VER} | Wine + Embeddable ZIP"
        echo ""
        echo "Kullanım: $0 <komut>"
        echo ""
        echo "  setup    Wine + Windows Python kur"
        echo "  install  Kütüphaneleri kur (numpy, scipy, cvxpy, pygame...)"
        echo "  build    .exe oluştur"
        echo "  test     Kurulumu kontrol et"
        echo "  clean    Build dosyalarını temizle"
        echo "  nuke     Wine Python'u tamamen kaldır"
        echo ""
        echo "Sıralama: setup → install → build"
        ;;
esac
