# Using Spack to Quickly Set Up Development Environment

For newly joined developers, can quickly set up development environment according to the following steps.

## 1. Local Quick Build

### 1. Local Quick Set Up Environment and Build

#### Method 1: Run One-click Script for Quick Build

prepare_cann_env.sh will one-click set up Spack environment and install all dependencies of ops-math

```bash
# Enter code root directory
cd ${local_repo_path}/ops-math
source spack/prepare_cann_env.sh
```

#### Method 2: Manual Environment Configuration (Suitable for Advanced Users or Custom Configuration)

If you want more fine-grained control over environment configuration, can manually configure according to the following steps:

##### Step 1: Download and Install Spack (Skip if Spack is already installed)

```bash
# Select installation directory (default is $HOME)
export SPACK_INSTALL_DIR="$HOME"
# Download Spack v1.1.0 and activate Spack tool
cd $SPACK_INSTALL_DIR
git clone https://gitcode.com/GitHub_Trending/sp/spack.git -b v1.1.0 --depth=2
source $SPACK_INSTALL_DIR/spack/share/spack/setup-env.sh
# Verify Spack version after installation
spack --version
```

##### Step 2: Set Spack Package Default Installation Path

Spack package default installation path is in $SPACK_INSTALL_DIR/spack/opt, can modify through the following command:

```bash
# Regular user
spack config --scope user add "config:install_tree:root:$HOME/.spack"
# root administrator
spack config --scope user add "config:install_tree:root:/opt/spack"
```

Regular users are recommended to use directory under personal home to avoid being modified by other users causing environment instability.
Administrators are recommended to use global path, convenient to use [Spack chain capability](https://spack.readthedocs.io/en/latest/chain.html) to uniformly install software for other users:

##### Step 3: Add External Installed Tools to Spack Environment

```bash
spack compiler find  # Configure locally installed compilers, such as gcc
```

##### Step 4: Configure Spack Official Repository GitCode Mirror Source

Spack default official repository address is [github address](https://github.com/spack/spack-packages.git), can modify default official repository through the following way, using GitCode mirror source to accelerate:
Modify repos.yaml under ~/.spack directory, if not exist create this file, write the following content

```yaml
repos:
  # Spack built-in repository
  builtin:
    git: https://gitcode.com/spack/spack-packages.git
    branch: releases/v2025.11
```

Verify repository configuration: `spack repo list`

##### Step 5: Download and Add CANN Community Spack Package Repository

```bash
# Clone CANN Spack package repository
cd $SPACK_INSTALL_DIR
git clone --depth=1 https://gitcode.com/cann/cann-spack-package.git

# Add to Spack repository list
spack repo add $SPACK_INSTALL_DIR/cann-spack-package

# Verify successful addition
spack repo list | grep cann
```

##### Step 6: Create and Activate Spack Environment

```bash
# Create environment named cann-dev-env
spack env create cann-dev-env ${local_repo_path}/ops-math/spack/spack.yaml

# Activate environment
spack env activate cann-dev-env

# Verify environment activation
spack env status
```

##### Step 7: Set Terminal Auto Load (Optional)

```bash
# Add Spack environment configuration to .bashrc
echo "source $SPACK_INSTALL_DIR/spack/share/spack/setup-env.sh" >> ~/.bashrc

# Take effect immediately
source ~/.bashrc
```

##### Step 8: Configure Development Mode and Install

```bash
# Set ops-math to development mode (point to local code)
spack develop -p ${local_repo_path}/ops-math cann-ops-math@master

# Add package to environment (adjust variants as needed)
spack add cann-ops-math@master+pkg+jit

# Resolve dependencies
spack concretize -f

# Install
spack install
```

### 2. View Artifact Location

Execute command

```bash
spack location -i cann-ops-math
```

Can view ops-math compilation generated run package location, run package has been automatically installed to ASCEND_HOME_PATH

## 2. Rebuild After Modifying Code

### 1. Uninstall Built Artifacts

```bash
spack uninstall -y cann-ops-math
```

### 2. Remove Existing Variants and Add New Variants

If want to specify different build parameters, can achieve through replacing Spack package variants
(Not changing build parameters, can skip this step)

```bash
spack change cann-ops-math@master +pkg +jit soc=ascend910b # Example
```

### 3. Re-resolve Dependencies

```bash
spack concretize -f
```

### 4. Rebuild

```bash
spack install
```

## 3. Re-enter Spack Environment

Opening new terminal needs to re-enter Spack environment

```bash
spack env activate cann-dev-env
```

## 4. Clean Environment or Uninstall Spack

If don't want to continue using Spack, can use `prepare_cann_env.sh` clean parameter to uninstall Spack and clean environment variables through **restarting terminal**

```bash
source prepare_cann_env.sh clean
```

## 5. ops-math Spack Build Variants

Through `spack info cann-ops-math` command can view parameters supported by ops-math, specific variant meanings please refer to [build.md](./build.md#parameter-description).

## 6. Spack Development and Debugging Command Guide

```shell
# View current environment list: If generated from one-click script prepare_cann_env.sh, development environment cann-dev-env is ready
spack env list

# Create environment: Environments managed by Spack are isolated from each other, linked to software packages in link form
spack env create <env-name>

# Enable environment: Currently activated environment font is green, Spack commands are based on currently activated environment, unrelated to location
spack env activate <env-name>

# View which software packages are introduced in current environment: If executed not in specific environment, view all environments
spack find                              # List all software packages
spack find -L                           # Display full hash
spack find <package-name>               # Filter by package name, if not provided then all packages
spack find --deps <package-name>        # Display dependency tree of this package
spack find --explicit <package-name>    # Display manually installed top-level packages Abbreviation: -e
spack find -p <package-name>            # Display full installation path
spack find -lv <package-name>           # Display full hash and variant information

# Add software package to current environment:
spack add <package-name>                # Also supports spec syntax, specify version, variant, compiler, etc.

# Remove software package from current environment: Spack will not delete software package, only remove dependency between current environment and software package
spack remove <package-name>

# Uninstall software package: Need all Spack environments not to depend on this software package to be uninstalled by Spack, will uninstall locally installed packages, use with caution
spack uninstall <package-name>

# Delete a Spack environment: Physically delete files, use with caution
spack env remove <env-name>

# Deactivate current environment:
spack env deactivate

# Search software packages supported by Spack
spack list <package-name>
spack list 'py-*'        # List all Python packages
spack list -d 'mpi'      # Search mpi in description

# View available version numbers
spack versions <package-name>

# Clear Spack build cache:
spack clean
spack clean -all   # Clear all old source code and old build records and cache, use with caution

# View current software package information: View all supported versions of this software package, Spack default tends to latest version
spack info <package-name>

# Install software package, Spack automatically selects latest version
spack install                                                   # Install all uninstalled software packages in current environment
spack install <package-name>                                    # Specify specific software package name
spack install <package-name>@1.12.0%gcc@11.4.0                  # spec syntax, specify version and compiler version
spack install <package-name>+mpi+fortran ^openmpi@4.1           # spec syntax, specify variants
spack install <package-name1> <package-name2> <package-name3>   # Install multiple packages simultaneously
spack -k install                                                # Skip security authentication during installation, use with caution
spack install --verbose                                         # Output detailed build log, abbreviation: -v
spack install --no-checksum                                     # Disable network checksum verification during installation, use with caution

# Concretize current environment: Spack's strategy is to default to reusing software packages already installed locally, even if higher version exists and meets dependency conditions, unless local package no longer meets dependency conditions
spack concretize

# Download software package source code from remote to local and register as development package: Otherwise Spack referenced remote files will be treated as cache temporary files, will be cleared by spack clean
spack develop <package-name>

# Register development package from local code development directory
git clone <git-path>  # Clone source code and enter directory
cd /package/path
spack develop --no-clone <package-name>

# Resolve software package dependency information:
spack spec <package-name>

# If git repository has updates, this command can synchronize updates according to remote repository address and branch configured in repos.yaml
spack repo update <repo name>

# Locate software package installation location:
spack location -b <package-name>

# View compiler configuration:
spack compilers                                                 # View all supported compilers, same as spack compiler list
spack compiler add /path/to/compiler/bin                        # Manually add compiler

# View locally installed software reused by Spack
spack external find

# View Spack version, verify installation:
spack --version

# Edit Spack configuration: Will automatically open vim
spack config edit config
spack config edit packages
```

## 7. One-click Script prepare_cann_env.sh Generated Paths (Default)

```shell
# CANN open source repository directory:
$HOME/cann-spack-package

# Spack managed software package installation directory:
root user: /opt/spack/linux-*
Regular user: $HOME/.spack/linux-*

# Spack configuration directory:
$HOME/.spack

# Spack provided builtin software package directory:
$HOME/.spack/package_repos/opgus4u/repos/spack_repo/builtin/packages

# Spack main directory:
$HOME/spack

# Spack environment directory:
$HOME/spack/var/spack/environments
```

## 8. FAQ

### 1. What to do if errors occur when executing prepare_cann_env.sh

If machine has installed Spack before, may cause environment variable or configuration file residue. After backing up key data, can execute ```source prepare_cann_env.sh clean``` and restart terminal then re-execute

### 2. What to do if ssl errors occur during installation

Such errors are caused by machine missing related certificates, please configure certificates yourself and re-run

### 3. How to uninstall if don't want to use Spack anymore

Execute ```source prepare_cann_env.sh clean``` can automatically uninstall Spack, root user also needs to execute ```rm -rf /opt/spack``` to uninstall all Spack software packages

## 9. For More Spack Operations Please Refer to Official Documentation

<https://spack.readthedocs.io/en/latest/>
