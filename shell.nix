{ pkgs ? import <nixpkgs> {} }:

(pkgs.mkShell.override { stdenv = pkgs.clangStdenv; }) {
  name = "kribu-dev-shell";

  # Allow compiling FetchContent/CPM dependencies located in the local build directory
  NIX_ENFORCE_PURITY = 0;

  # Allow compiler flags like -march=native and -mtune=native to optimize for this host machine
  NIX_ENFORCE_NO_NATIVE = 0;

  # Development tools for the repository and custom tools requested by user
  buildInputs = with pkgs; [
    fish
    asciinema
    atuin
    zoxide
    bat
    micro

    cmake
    ccache
    entr
    doxygen
    git
    gnumake
    uv
    python3
    duckdb
  ];

  shellHook = ''
    export LD_LIBRARY_PATH="${pkgs.lib.makeLibraryPath [ pkgs.stdenv.cc.cc.lib ]}:''${LD_LIBRARY_PATH:-}"

    echo "===================================================="
    echo " Welcome to Kribu Dev Shell!"
    echo "===================================================="
    
    # Enter fish shell
    exec fish
  '';
}
