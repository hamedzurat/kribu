{ pkgs ? import <nixpkgs> {} }:

pkgs.mkShell {
  name = "kribu-dev-shell";

  # Development tools for the repository and custom tools requested by user
  buildInputs = with pkgs; [
    # Custom shell & tools requested by the user
    fish
    asciinema
    atuin
    zoxide       # Modern Rust-based directory jumper matching 'z'
    bat
    micro

    # Repository build tools
    clang
    cmake
    ccache
    entr
    doxygen
    git
    gnumake
    uv
    python3
    arrow-cpp
  ];

  shellHook = ''
    # Environment variables or alias configurations can go here
    export CC=clang
    export CXX=clang++

    echo "===================================================="
    echo " Welcome to Kribu Dev Shell!"
    echo " - Requested tools loaded: fish, asciinema, atuin, zoxide (z), bat, micro"
    echo " - Dev tools loaded: clang, cmake, ccache, entr, doxygen, uv, python3"
    echo " Starting Fish shell..."
    echo "===================================================="
    
    # Enter fish shell
    exec fish
  '';
}
