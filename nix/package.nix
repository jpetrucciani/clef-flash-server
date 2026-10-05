{ callPackage, lib, stdenvNoCC, fetchurl, makeWrapper, clang, python312, python313, python314, uv-nix, workspaceRoot ? ../., pythonVersion ? "3.13", isWSL ? false }:
let
  python = {
    "3.12" = python312;
    "3.13" = python313;
    "3.14" = python314;
  }.${pythonVersion} or (throw "Clef supports Python 3.12, 3.13, or 3.14; got ${pythonVersion}");
  # fetchTarball returns a realised store path string; filesets require a path.
  sourceRoot = /. + builtins.unsafeDiscardStringContext (toString workspaceRoot);
  version = (lib.importTOML (sourceRoot + "/pyproject.toml")).project.version;
  modelRevision = "17f0b0ad64efb65d273590632833508766b2aae6";
  jointSchemaModel = fetchurl {
    url = "https://huggingface.co/Cloudflare/clef-flash/resolve/${modelRevision}/joint_schema_model.py";
    hash = "sha256-DjBM98ZQDou1m+9+Kv0sY3P4JZbfs7V9Gqk8F14tw6M=";
  };
  uvEnv = uv-nix.mkEnv {
    name = "clef-flash-server";
    inherit python;
    workspaceRoot = lib.fileset.toSource {
      root = sourceRoot;
      fileset = lib.fileset.unions [
        (sourceRoot + "/pyproject.toml")
        (sourceRoot + "/uv.lock")
        (lib.fileset.fileFilter (file: file.hasExt "py") (sourceRoot + "/clef_flash_server"))
      ];
    };
    gitignore = false;
    enableCuda = true;
    pyprojectOverrides = final: prev: {
      clef-flash-server = prev.clef-flash-server.overrideAttrs (old: {
        postInstall = (old.postInstall or "") + ''
          cp ${jointSchemaModel} "$out/${python.sitePackages}/cloudflare_clef_release.py"
        '';
      });
      bitsandbytes = prev.bitsandbytes.overrideAttrs (old: {
        # This package serves NVIDIA CUDA 13 only. Recent wheels also contain
        # AMD, Intel, and older CUDA backends whose runtimes are not dependencies.
        postInstall = (old.postInstall or "") + ''
          rm -f "$out/${python.sitePackages}/bitsandbytes"/libbitsandbytes_rocm*.so \
            "$out/${python.sitePackages}/bitsandbytes"/libbitsandbytes_xpu*.so \
            "$out/${python.sitePackages}/bitsandbytes"/libbitsandbytes_cuda11*.so \
            "$out/${python.sitePackages}/bitsandbytes"/libbitsandbytes_cuda12*.so
        '';
        # Keep the CUDA 13 kernel's dependencies aligned with PyTorch's locked
        # NVIDIA wheels.
        buildInputs = (old.buildInputs or [ ]) ++ [
          final.nvidia-cuda-runtime
          final.nvidia-cublas
          final.nvidia-cusparse
        ];
        preFixup = (old.preFixup or "") + ''
          addAutoPatchelfSearchPath "${final.nvidia-cuda-runtime}"
          addAutoPatchelfSearchPath "${final.nvidia-cublas}"
          addAutoPatchelfSearchPath "${final.nvidia-cusparse}"
        '';
      });
    };
  };
  driverPath = if isWSL then "/usr/lib/wsl/lib" else "/run/opengl-driver/lib";
in
stdenvNoCC.mkDerivation {
  pname = "clef-flash-server";
  inherit version;
  dontUnpack = true;
  nativeBuildInputs = [ makeWrapper ];
  installPhase = ''
    runHook preInstall
    mkdir -p $out/bin
    sitePackages="${uvEnv}/${python.sitePackages}"
    wheelCudaLibs="$sitePackages/torch/lib"
    for libdir in "$sitePackages"/nvidia/*/lib; do
      wheelCudaLibs="$wheelCudaLibs:$libdir"
    done
    for program in clef-flash-server python; do
      wrapperName="$program"
      if [[ "$program" == python ]]; then
        wrapperName=clef-flash-python
      fi
      makeWrapper "${uvEnv}/bin/$program" "$out/bin/$wrapperName" \
        --prefix LD_LIBRARY_PATH : "$wheelCudaLibs:${driverPath}" \
        --prefix LIBRARY_PATH : "${driverPath}" \
        --set TRITON_LIBCUDA_PATH "${driverPath}" \
        --set CC "${clang}/bin/clang" \
        --prefix PATH : ${lib.makeBinPath [ clang ]}
    done
    makeWrapper ${uvEnv}/bin/hf $out/bin/clef-flash-download \
      --add-flags "download Cloudflare/clef-flash --revision ${modelRevision}"
    runHook postInstall
  '';
  passthru = {
    inherit modelRevision uvEnv python pythonVersion;
    wsl = callPackage ./package.nix { inherit workspaceRoot pythonVersion; isWSL = true; };
    python313 = callPackage ./package.nix { inherit workspaceRoot isWSL; pythonVersion = "3.13"; };
    python314 = callPackage ./package.nix { inherit workspaceRoot isWSL; pythonVersion = "3.14"; };
  };
  meta = {
    description = "NF4 CUDA server for Cloudflare Clef-Flash's SystemOne decision API";
    homepage = "https://github.com/jpetrucciani/clef-flash-server";
    license = lib.licenses.asl20;
    maintainers = with lib.maintainers; [ jpetrucciani ];
    mainProgram = "clef-flash-server";
    platforms = [ "x86_64-linux" ];
    skipBuild = true;
  };
}
