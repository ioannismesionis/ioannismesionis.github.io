---
layout: post
title: An R-tistict Application
subtitle: R Shiny Template
image: /img/r-shiny-template/r.png
bigimg: /img/r-shiny-template/analytics-r-shiny.jpg
tags: [R, shiny, template]
---

Shiny is an R package that makes it easy to build interactive web apps straight from R. You can host standalone apps on a webpage or embed them in R Markdown documents or build dashboards. You can also extend your Shiny apps with CSS themes, htmlwidgets, and JavaScript actions.

Shiny combines the computational power of R with the interactivity of the modern web.

The purpose of this post is to present a minimalistic R shiny app I developed some time ago. The full code is present on my [GitHub Repository](https://github.com/ioannismesionis/r-shiny-template "GitHub Repository").

For sample images, please see below.

R Shiny Template
==========================

The welcome page of the app let's you upload a file. 

![](/img/r-shiny-template/upload.png)

For example, a .csv file can be uploaded through your local computer and feed the following pages.

1. Results tab
2. Plot tab

Both in these tabs, the user is given the option to download the data in a .csv format by pressing the download button as shown below.

![](/img/r-shiny-template/results.png)

The plot tab allows of data visualisation which can be hardcored in the respective plot-tab.R file.

Finally, the about tab contains information about the development of the R Shiny and references that were used as well.

![](/img/r-shiny-template/about.png)

The present R Shiny template, is accompanied by a custom .css file that further decorates the app. For example, see the underline under the tabs names.

![](/img/r-shiny-template/underline.png)

Technical side
==========
To start the app, open the `ui.R` script and press the `Run App` button shown on the upper right corner of the R script.

To build on top of the template, you need to change the respective `tab-*.R` scripts both in the `ui` and `server` folders.

### _ui_ folder:
The `tab-*.R` scripts contain the _ui_ elements displayed in the app using the built-in function of R Shiny. Some of the functions used are the following:
  - `fileInput()`
  - `uiOutput()`
  - `downloadButton()`
 
The aforementioned functions create ui elements shown in the app but the functionality of these is specified in the _server_ folder in the respective `tab-*.R` script.

E.g. The `downloadButton()` function specified in the _`ui/tab-upload.R`_  displays a download button in the frontend of the app. However, the action that occurs when pressing the button is to be specified in the respective _`server/tab-upload.R`_ in the _server_ folder.

In this folder, you can also change the name of the tabs or add extra tabs however it seems fit for the purpose of the shiny app needed to be developed. The code is commented to assist as much as possible to make such changes as easy as possible. 

### _server_ folder:

In the server folder, the functionality of the respective tabs is specified. The trickiest part of R-Shiny is the reactivity functions that make the code reactive triggered by specific events.

Some of the functions that are being used are the following:

  - `observeEvent()`
  - `reactive()`
  - `reactiveVal()`
 
or for rendering the ui elements

  - `renderUI()`
  - `renderPlot()`
  - `renderDT()`
 
To know more about R-Shiny and reactivity, feel free to address to the following [link](https://shiny.rstudio.com/tutorial/). 
  
## References:
I used many tricks suggested by Dean Attali.

Please check his GitHub account for more:
https://github.com/daattali
