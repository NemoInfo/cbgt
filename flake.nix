{
  description = "CBGT flake pyhton3 + rust";

  inputs = { nixpkgs.url = "github:nixos/nixpkgs?ref=nixos-unstable"; };

  outputs = { self, nixpkgs }:
    let
      system = "x86_64-linux";
      pkgs = nixpkgs.legacyPackages.${system};
    in {
      devShells.${system}.default = pkgs.mkShell {
        buildInputs = with pkgs.python312Packages; [
          pkgs.jupyter-all
          pkgs.maturin
          pkgs.virtualenv
          scipy
          python
          numpy
          pandas
          matplotlib
          numba
          tqdm
          marimo
          polars
          altair

          (pkgs.rustc) # Rust compiler
          (pkgs.cargo) # Rust dependecy manager
        ];

        shellHook = ''
          source venv/bin/activate
          THEME="af-magic" exec $SHELL
        '';
      };
    };
}
