{
   description = "Exo";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-23.05";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem
      (system:
      let
        python = "python39";
        version = builtins.substring 0 8 self.lastModifiedDate;

        pkgs = import nixpkgs {
          inherit system;
        };

        ########## Extra Packages not in NixPkgs ###########
        packagesExtra = rec {

          asdl = pkgs.python39Packages.buildPythonPackage rec {
            pname = "asdl";
            version = "2018.01.10";
            src = builtins.fetchGit{
              url = "https://github.com/fpoli/python-asdl";
              ref = "master";
              rev = "0cf2a6c7863700189d5d9ce84b4f960c567cf0ee";
            };
            doCheck = false; 
            pythonImportsCheck = [ "asdl" ];
          };
          
          #Note: DOES NOT work with 0.2.0
          asdl-adt = pkgs.python39Packages.buildPythonPackage rec {
            pname = "asdl-adt";
            version = "0.1.0";
            src = builtins.fetchGit{
              url = "https://github.com/ChezJrk/asdl";
              ref = "master";
              rev = "7bbfa403b05d0b8bb32f03c287f7f00c1f2e031d";
            };
            propagatedBuildInputs = with pkgs.python39Packages; [
              setuptools
              attrs
              asdl
            ];
            doCheck = false; 
            format = "pyproject";
            pythonImportsCheck = [ "asdl_adt" ];
          };

          exo = pkgs.python39Packages.buildPythonPackage rec {
            pname = "exo";
            inherit version;
            src = ./.;
            #somehow there are weird issues with the name z3... it's definitely installed though
            prePatch = ''
                substituteInPlace setup.cfg \
                  --replace "z3-solver>=4.8.12.0" ""
            '';

            propagatedBuildInputs = with pkgs.python39Packages; [
              setuptools
              z3
              pysmt
              astor
              numpy
              yapf
            ] ++ [ asdl asdl-adt ];
            doCheck = false; 
            pythonImportsCheck = [ "exo" ];
          };

        };

        ########## Python Definitions ###########

        pydeps = ( with packagesExtra; [
            asdl
            asdl-adt
            exo
          ]) ++ ( with pkgs.python39Packages; [
            pytest
            pysmt
            astor
            numpy
            yapf
            z3
          ]);

        fullPython = ( pkgs.python39.buildEnv.override {
          extraLibs = pydeps;
        });

      in
      rec {

        #Main environment for use of GPTune, with the correct python packages preloaded
        devShell = pkgs.mkShell {
          nativeBuildInputs = [
            fullPython
            pkgs.cmake
            pkgs.ninja
          ];
        };

        defaultPackage = devShell;
      });
}
