{ pkgs ? import <nixpkgs> {} }:

pkgs.mkShell {
  name = "kribu-dev-shell";

  # Development tools for the repository and custom tools requested by user
  buildInputs = with pkgs; [
    fish
    asciinema
    atuin
    zoxide
    bat
    micro

    clang
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
    # Environment variables or alias configurations can go here
    export CC=clang
    export CXX=clang++

    echo "===================================================="
    echo " Welcome to Kribu Dev Shell!"
    echo "===================================================="
    
    # Enter fish shell
    exec fish
  '';
}
