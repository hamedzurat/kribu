{ pkgs ? import <nixpkgs> {} }:

(pkgs.mkShell.override { stdenv = pkgs.clangStdenv; }) {
  name = "kribu-dev-shell";

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
    echo "===================================================="
    echo " Welcome to Kribu Dev Shell!"
    echo "===================================================="
    
    # Enter fish shell
    exec fish
  '';
}

