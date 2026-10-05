{ pkgs ? import
    (fetchTarball {
      # nixup: pin=jpetrucciani/nix;
      name = "jpetrucciani-2026-10-02";
      url = "https://github.com/jpetrucciani/nix/archive/0ccc5a242070b7c41ab6b5be6504d07beb1bcd3a.tar.gz";
      sha256 = "0win1p03zld6vn8vis1b7f5yvinbwyylxm0s54mggwm3xn77zadx";
    })
    { }
, isWSL ? builtins.pathExists /usr/lib/wsl/lib/libcuda.so.1
, pythonVersion ? "3.13"
}:
let
  name = "clef-flash-server";
  package = pkgs.callPackage ./nix/package.nix { workspaceRoot = ./.; inherit pythonVersion; };
  runtimePackage = if isWSL then package.wsl else package;
  driverPath = if isWSL then "/usr/lib/wsl/lib" else "/run/opengl-driver/lib";
  sitePackages = "${package.uvEnv}/${package.python.sitePackages}";

  scripts = {
    test = pkgs.pog {
      name = "test-clef-flash";
      description = "Run schema tests and real CUDA tests when CLEF_TEST_MODEL_PATH is set";
      runtimeInputs = [ runtimePackage ];
      script = ''
        export PYTHONDONTWRITEBYTECODE=1
        clef-flash-python -m unittest discover -s tests -v "$@"
      '';
    };
  };
  tools = with pkgs; {
    cli = [ clang jfmt pyright ruff uv ];
    python = [ runtimePackage package.uvEnv ];
    scripts = lib.attrValues scripts;
  };
in
(pkgs.mkShellNoCC (package.uvEnv.uvEnvVars // {
  inherit name;
  packages = pkgs.lib.flatten (builtins.attrValues tools);
  CC = "${pkgs.clang}/bin/clang";
  TRITON_LIBCUDA_PATH = driverPath;
  CLEF_MODEL_REVISION = package.modelRevision;
  CLEF_PYTHON_VERSION = pythonVersion;
  shellHook = ''
    wheel_cuda_libs="${sitePackages}/torch/lib"
    for libdir in "${sitePackages}"/nvidia/*/lib; do
      wheel_cuda_libs="$wheel_cuda_libs:$libdir"
    done
    export LD_LIBRARY_PATH="$wheel_cuda_libs:${driverPath}''${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    export LIBRARY_PATH="${driverPath}''${LIBRARY_PATH:+:$LIBRARY_PATH}"
    export CLEF_MODEL_PATH="''${CLEF_MODEL_PATH:-''${XDG_CACHE_HOME:-$HOME/.cache}/clef-flash/models/$CLEF_MODEL_REVISION}"
    export PYTHONDONTWRITEBYTECODE=1
    unset wheel_cuda_libs libdir
  '';
})) // { inherit package scripts; }
