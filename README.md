[![Build Status](https://github.com/I2PC/xmipp/actions/workflows/build.yml/badge.svg)](https://github.com/I2PC/xmipp/actions/workflows/build.yml)

# Xmipp
<a href="https://github.com/I2PC/xmipp"><img src="https://github.com/I2PC/scipion-em-xmipp/blob/devel/xmipp3/xmipp_logo.png" alt="drawing" width="330" align="left"></a>


  
Xmipp is a suite of image processing programs, primarily aimed at single-particle 3D electron microscopy, designed and managed by the [Biocomputing Unit](http://biocomputingunit.es/) located in Madrid, Spain.

The Xmipp project is divided into four repositories. 
This is the main repository, which contains the majority of the source code for the programs, additional scripts, and tests. Three remaining repositories, [XmippCore](https://github.com/I2PC/xmippCore/), [XmippViz](https://github.com/I2PC/xmippViz/), and [Scipion-em-xmipp](https://github.com/I2PC/scipion-em-xmipp), are automatically downloaded during the installation process.


## Getting started
To have a complete overview about Xmipp please visit the [documentation web](https://i2pc.github.io/docs/). The recommended way for users (not developers) to install and use Xmipp is via the [Scipion](http://scipion.i2pc.es/) framework, where you can use Xmipp with other Cryo-EM-related software. 



### Installation Telemetry

Xmipp collects **basic information about your installation environment**  (such as library versions, system architecture, and operating system) to  improve compatibility, performance, and stability.

If you prefer **not to send this data**, you can disable collection by setting: `SEND_INSTALLATION_STATISTICS=OFF`
For more information, please visit [Data collection](https://i2pc.github.io/docs/Others/Enhancing/index.html#data-collection)


### Developer installation with Scipion
For developers, it becomes handy to install Xmipp in an external directory for easy integration with IDEs. By default Xmipp installer takes care of linking to Scipion.

The first step is to download Xmipp sources:

`git clone https://github.com/I2PC/xmipp.git xmipp-bundle & cd xmipp-bundle`

If you want to checkout an scpecific branch use the following command. For repositories where the branch does not exist, devel is utilized.

`./xmipp getSources -b branch`

Compile Xmipp

`scipion3 run ./xmipp`

Refer to `./xmipp --help` for additional info on the compilation process and possible customizations.

Install the Scipion plugin.

`scipion3 installp -p src/scipion-em-xmipp --devel`

## That's all

If you miss some feature or find a bug, please [create an issue](https://github.com/I2PC/xmipp/issues/new) and we will have a look at it, you can also contact us by xmipp@cnb.csic.es or Discord (Scipion chanel). For more details, troubleshootings and information visit the [documentation web.]([https://github.com/I2PC/xmipp/wiki](https://i2pc.github.io/docs/index.html))

If you like this project, Watch or Star it! We are also looking for new collaborators, so contact us if you’re interested
