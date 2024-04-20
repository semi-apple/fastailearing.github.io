<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>For Assignment 2</title>
</head>
<body>
    <h1 style="text-align: center;">Feedbacks about FastAI</h1>
    <div class="centered-div">
        <p>In this assignment, we focus on using FastAI to tackle image recognition problems. We will employ tools such as DuckDuckGo, FastAI, FastDownload, and PIL to gather image data and train neural network models. </p>

        <p>The FastAI course is practice-oriented, encouraging students to deepen their understanding of deep learning through hands-on practice. These courses usually cover the entire process from data preprocessing to model training, evaluation, and deployment. Additionally, the FastAI community is active and provides a wealth of support, offering numerous resources and assistance to learners such as fastai (<a href="https://course.fast.ai/">https://course.fast.ai/</a>). With FastAI, we can not only learn the latest deep learning technologies but also acquire the ability to apply these technologies to solve real-world problems.</p>

        <h2>Is it a bird or not</h2>
        <p>This course was quite novel to me. I originally thought that to train a neural network capable of recognizing images, one would need to build a database locally and then manually construct neural networks such as CNNs, train, and test them. This process would require writing a lot of code, with even more time spent debugging them. However, the FastAI library has incorporated APIs for various complex neural networks, and we only need to call the API that suits our needs. For me, more time has shifted towards researching how to use these libraries. Indeed, engineers spend most of their time “Google-Oriented Programming”.</p>

        <p>Take the ResNet18 used in the assignment as an example. ResNet18 is a widely used convolutional neural network favored for its good performance and relatively quick training speed.</p>

        <p>Using FastAI's DataBlock API to construct data loaders (DataLoaders) for image classification involves automating the loading, splitting, and transformation of data. In this example, the training set comprises 80% of the data, with the remaining 20% used for validation. Additionally, the default batch size is set to 64, which specifies the number of images processed by the network in each training iteration.</p>

        <pre class="code-block">
            <code class="code-line">dls = DataBlock(</code>
            <code class="code-line">    blocks=(ImageBlock, CategoryBlock),</code>
            <code class="code-line">    get_items=get_image_files,</code>
            <code class="code-line">    splitter=RandomSplitter(valid_pct=0.2, seed=42),</code>
            <code class="code-line">    get_y=parent_label,</code>
            <code class="code-line">    item_tfms=[Resize(192, method='squish')]</code>
            <code class="code-line">).dataloaders(path)</code>
            <code class="code-line">dls.show_batch(max_n=6)</code>
        </pre>
        <p class="subtitle">Code example of DataBlock</p>

        <div class="img-div"><img src="../pictures/DataBlock-output.png" alt="Output" width="60%" height="auto"></div>
        <p class="subtitle">Output</p>

        <p>FastAI facilitates image transformations such as scaling and resizing to process input images, ensuring that the model can learn from standardized inputs. Using FastAI's vision_learner, the ResNet18 network is loaded, and FastAI's fine_tune method is utilized to adjust the pretrained model to adapt to a new dataset (in this case, distinguishing between "bird" and "non-bird" images).</p>

        <pre class="code-block">
            <code class="code-line">learn = vision_learner(dls, resnet18, metrics=error_rate)</code>
            <code class="code-line">learn.fine_tune(3)</code>
        </pre>
        <p class="subtitle">Code example of ResNet18</p>

        <div class="img-div"><img src="../pictures/ResNet-output.png" alt="Output" width="70%" height="auto"></div>
        <p class="subtitle">Output</p>

        <p>Fine-tuning is a process that adapts a pretrained model to a new task by making minor adjustments to the existing model weights. After training, the model's performance is evaluated using a validation set, and new images are predicted to determine their categories.</p>

        <pre class="code-block">
            <code class="code-line">is_bird,_,probs = learn.predict(PILImage.create('bird.jpg'))</code>
            <code class="code-line">print(f"This is a: {is_bird}.")</code>
            <code class="code-line">print(f"Probability it's a bird: {probs[0]:.4f}")</code>
        </pre>
        <p class="subtitle">Test the model</p>

        <div class="img-div"><img src="../pictures/test-output.png" alt="Output" width="50%" height="auto"></div>
        <p class="subtitle">Output</p>

        <p>This case excellently demonstrates how tools like FastAI can be used to quickly build a practical image classification model from scratch, achieving high accuracy through the use of pretrained networks and fine-tuning. These steps and methods provide an effective blueprint for the rapid development and deployment of machine learning models.</p>

        <h2>Challenges While completing the task </h2>
        <p>Next, I would like to discuss the difficulties I encountered while learning FastAI and the solutions.</p>

        <h3>Docker Configuration</h3>
        <p>Since the course22 file needed to run in a Docker environment, we first had to set up the Docker environment. Fortunately, Prof. Lovell had already written a blog post on this topic, which allowed us to complete the configuration without spending too much time.</p>

        <h3>GPU initialization</h3>
        <p>During the environment setup process, we needed to update our GPU, but often I would see the WDDM (Windows Display Driver Model) label, indicating that the GPU was not accessible, and the version of GPU driver in Windows is different with the version in WSL</p>

        <div class="img-div"><img src="../pictures/GPU-output-in-wsl.png" alt="Output" width="70%" height="auto"></div>
        <p class="subtitle">Terminal in WSL</p>
        <div class="img-div"><img src="../pictures/GPU-output-in-Windows.png" alt="Output" width="70%" height="auto"></div>
        <p class="subtitle">Terminal in Windows</p>

        <p>Here we have two different versions of GPU. In Windows, the version is NVIDIA-SMI 536.19, yet in WSL is NVIDIA-SMI 535.54.02. I notice that the GPU driver in Ubuntu 22.04 may be different with the GPU driver in Windows, even though I am so sure that I have installed only one version's GPU driver. The driver which has a low version cannot run on Windows, but can run on Ubuntu. The driver which has a high version can run on Windows, yet not on Ubuntu.</p>

        <p>On this assignment, Ubuntu 22.04 only support the GPU driver version lower that 537 (based on website). Once I have installed a high version of GPU and it cannot be displayed on Ubuntu, I have no choice but uninstalled the higher version and reinstalled a lower version.</p>

        <h3>Red C disk</h3>
        <p>After fixing GPU issue, I started to run the code. The situation was good at the beginning, but suddenly my computer became very slow. It turns out that my C driver was filled, and there is no space for this project.</p>
        <div class="img-div"><img src="../pictures/red-C.png" alt="Angry C disk" width="50%" height="auto"></div>
        <p class="subtitle">Angry C disk</p>

        <p>I asked this problem on ED for some help, and Prof. Lovell said I can change 'Image path' in Docker. </p>
        <div class="img-div"><img src="../pictures/set-image-path.png" alt="Set image path in Docker" width="70%" height="auto"></div>
        <p class="subtitle">Set image path in Docker</p>

        <p>After setting the path, the issued was solved, but there are other problems that bother me, so I have to use the laboratory computer with a remote connection. It turns out that the computer in the laboratory is still very reliable.</p>

        <h3>Export .ipynb to .pdf</h3>
        <p>After finishing all of the problems related to this project, I was trying to export 'ipynb' to 'pdf'. But it seems that the laboratory computer has some kind of protection that does not allow us to export files directly from WSL to the desktop, even though I have set the correct UNCHost.</p>

        <p>Finally, it suddenly occurred to me that since the code is running in Docker, maybe the file is also can be found in docker. Following this thought, I browsed the folder in image and found the file I want. </p>

        <div class="img-div"><img src="../pictures/Files-in-docker.png" alt="Files in Docker" width="70%" height="auto"></div>
        <p class="subtitle">Files in Docker</p>

        <p>The above is my impression of learning FastAI; it is an excellent course. Most of the content in the course, such as Docker, Linux command line, and WSL, comprises what I am eager to learn. These technologies will not only enhance my competitiveness but also allow me to be exposed to more cutting-edge related technologies and broaden my horizons.</p>
    </div>

    <footer>
        <p>Kun Zhou | </p>
        <p>20-04-2024 | </p>
        <a href="#">kun.zhou@uq.net.au</a></li>
      </footer>

</body>
<style>
    body {
        margin: 10px 10px;
        border: 10px 10px;
    }

    .code-block {
        background-color: #333; 
        color: #f8f9fa;        
        padding: 10px;      
        border-radius: 5px;    
        font-family: monospace; 
        margin: 10px auto;     
        line-height: 9px;  
    }

    .code-line {
        display: block;              
    }

    .centered-div {
        width: 70%; 
        margin: 0 auto; 
        text-align: left; 
    }

    .img-div {
        text-align: center;
    }

    .subtitle {
        text-align: center;
        margin-top: -2px;
        font-size: 20px;
        color: #8f9193;
    }

    footer {
        text-align: center;    
        padding: 10px 0;       
        border-top: 1px solid #eaeaea;
      }
      
      footer p {
        display: inline;
        margin: 10px 0;        
      }  
      
      footer a {    
        color: black;
        text-decoration: none;  
      }
      
      footer a:hover {
        color: rgb(20, 102, 178);
        text-decoration: underline; 
      }

      p {
        font-size: 20px;
        line-height: 1.4;
      }
      
</style>
</html>
