// get the ninja-keys element
const ninja = document.querySelector('ninja-keys');

// add the home and posts menu items
ninja.data = [{
    id: "nav-about",
    title: "About",
    section: "Navigation",
    handler: () => {
      window.location.href = "/";
    },
  },{id: "nav-publications",
          title: "Publications",
          description: "Publications by categories in reverse chronological order (* = equal contributions).",
          section: "Navigation",
          handler: () => {
            window.location.href = "/publications/";
          },
        },{id: "nav-projects",
          title: "Projects",
          description: "Over the years, I&#39;ve had the privilege of being involved in a wide range of cool projects. Below I list some noteworthy ones that I have had the opportunity to work on.",
          section: "Navigation",
          handler: () => {
            window.location.href = "/projects/";
          },
        },{id: "nav-blog",
          title: "Blog",
          description: "",
          section: "Navigation",
          handler: () => {
            window.location.href = "/blog/";
          },
        },{id: "nav-curriculum-vitae",
          title: "Curriculum Vitae",
          description: "",
          section: "Navigation",
          handler: () => {
            window.location.href = "/cv/";
          },
        },{id: "post-foundation-models-for-sequential-decision-making",
        
          title: "Foundation Models for Sequential Decision-Making",
        
        description: "In this post, I&#39;m hoping to give some sort of timeline of advances made in fields like reinforcement learning and robotics that I believe could be important to know about or might be important to realize AI embodied in the real world.e will start by reviewing some classical papers that combine Transformers with reinforcement learning for multi-task control. Then, we will look at some more recent advances that are used in simulated open-ended environments, and from there on we will move on to advances in control for real-world robotics with robotics foundation models.",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blog/2024/foundation-model-decision-making/";
          
        },
      },{id: "post-cs-330-lecture-8-variational-inference",
        
          title: "CS-330 Lecture 8: Variational Inference",
        
        description: "This lecture is part of the CS-330 Deep Multi-Task and Meta Learning course, taught by Chelsea Finn in Fall 2023 at Stanford. This post will talk about variational inference, which is a way of approximating complex distributions through Bayesian inference. We will go from talking about latent variable models all the way to amortized variational inference!",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blog/2024/cs330-stanford-var-inf/";
          
        },
      },{id: "post-cs-330-lecture-7-unsupervised-pre-training-reconstruction-based-methods",
        
          title: "CS-330 Lecture 7: Unsupervised Pre-Training: Reconstruction-Based Methods",
        
        description: "This lecture is part of the CS-330 Deep Multi-Task and Meta Learning course, taught by Chelsea Finn in Fall 2023 at Stanford. The goal of this post is to introduce to widely-used methods for unsupervised pre-training, which is essential in many fields nowadays, most notably in the development of foundation models. We also introduce methods that help with efficient fine-tuning of pre-trained models!",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blog/2024/cs330-stanford-upt-rbm/";
          
        },
      },{id: "post-cs-330-lecture-6-unsupervised-pre-training-contrastive-learning",
        
          title: "CS-330 Lecture 6: Unsupervised Pre-Training: Contrastive Learning",
        
        description: "This lecture is part of the CS-330 Deep Multi-Task and Meta Learning course, taught by Chelsea Finn in Fall 2023 at Stanford. The goal of this lecture is to understand the intuition, design choices, and implementation of contrastive learning for unsupervised representation learning. We will also talk about the relationship between contrastive learning and meta learning!",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blog/2024/cs330-stanford-upt-fsl-cl/";
          
        },
      },{id: "post-cs-330-lecture-5-few-shot-learning-via-metric-learning",
        
          title: "CS-330 Lecture 5: Few-Shot Learning via Metric Learning",
        
        description: "This lecture is part of the CS-330 Deep Multi-Task and Meta Learning course, taught by Chelsea Finn in Fall 2023 at Stanford. The goal of this lecture is to to understand the third form of meta learning: non-parametric few-shot learning. We will also compare the three different methods of meta learning. Finally, we give practical examples of meta learning, in domains such as imitation learning, drug discovery, motion prediction, and language generation!",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blog/2024/cs330-stanford-fsl-ml/";
          
        },
      },{id: "post-cs-330-lecture-4-optimization-based-meta-learning",
        
          title: "CS-330 Lecture 4: Optimization-Based Meta-Learning",
        
        description: "This lecture is part of the CS-330 Deep Multi-Task and Meta Learning course, taught by Chelsea Finn in Fall 2023 at Stanford. The goal of this lecture is to understand the basics of optimization-based meta learning techniques. You will also learn about the trade-offs between black-box and optimization-based meta learning!",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blog/2024/cs330-stanford-obml/";
          
        },
      },{id: "post-cs-330-lecture-2-transfer-learning-and-meta-learning",
        
          title: "CS-330 Lecture 2: Transfer Learning and Meta-Learning",
        
        description: "This lecture is part of the CS-330 Deep Multi-Task and Meta Learning course, taught by Chelsea Finn in Fall 2023 at Stanford. The goal of this lecture is to learn how to transfer knowledge from one task to another, discuss what it means for two tasks to share a common structure, and start thinking about meta learning.",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blog/2024/cs330-stanford-tl-ml/";
          
        },
      },{id: "post-cs-330-lecture-3-black-box-meta-learning-amp-in-context-learning",
        
          title: "CS-330 Lecture 3: Black-Box Meta-Learning &amp; In-Context Learning",
        
        description: "This lecture is part of the CS-330 Deep Multi-Task and Meta Learning course, taught by Chelsea Finn in Fall 2023 at Stanford. The goal of this lecture is to learn how to implement black-box meta-learning techniques. We will also talk about a case study of GPT-3!",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blog/2024/cs330-stanford-bbml-icl/";
          
        },
      },{id: "post-cs-330-lecture-1-multi-task-learning",
        
          title: "CS-330 Lecture 1: Multi-Task Learning",
        
        description: "This is the first lecture of the CS-330 Deep Multi-Task and Meta Learning course, taught by Chelsea Finn in Fall 2023 at Stanford. The goal of this lecture is to understand the key design decisions when building multi-task learning systems.",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blog/2024/cs330-stanford-mtl/";
          
        },
      },{id: "post-cs-330-deep-multi-task-and-meta-learning-introduction",
        
          title: "CS-330: Deep Multi-Task and Meta Learning - Introduction",
        
        description: "I have been incredibly interested in the recent wave of multimodal foundation models, especially in robotics and sequential decision-making. Since I never had a formal introduction to this topic, I decided to audit the Deep Multi-Task and Meta Learning course, which is taught yearly by Chelsea Finn at Stanford. I will mainly document my takes on the lectures, hopefully making it a nice read for people who would like to learn more about this topic!",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blog/2024/cs330-stanford-introduction/";
          
        },
      },{id: "books-the-godfather",
          title: 'The Godfather',
          description: "",
          section: "Books",handler: () => {
              window.location.href = "/books/the_godfather/";
            },},{id: "news-i-ve-completed-the-basis-of-this-website-feel-free-to-fork-the-template-from-my-repository-if-you-want-to-start-using-it",
          title: 'I’ve completed the basis of this website! Feel free to fork the template...',
          description: "",
          section: "News",},{id: "news-for-the-ml4science-project-at-epfl-we-are-hosted-by-volvo-group-trucks-technology-and-chalmers-university-of-technology-to-work-on-a-project-about-traffic-flow-prediction-for-autonomous-driving",
          title: 'For the ML4Science project at EPFL, we are hosted by Volvo Group Trucks...',
          description: "",
          section: "News",},{id: "news-i-am-honoured-to-announce-that-i-have-been-awarded-the-best-bachelor-s-thesis-award-2022-for-the-research-done-during-my-bachelor-s-at-maastricht-university-in-case-you-are-interested-in-reading-more-about-it-the-preprint-can-be-found-here",
          title: 'I am honoured to announce that I have been awarded the Best Bachelor’s...',
          description: "",
          section: "News",},{id: "news-i-am-happy-to-share-that-i-will-be-joining-the-laboratory-for-information-and-inference-systems-to-work-on-a-research-project-during-the-spring-2023-semester-my-work-supervised-by-grigorios-chrysos-and-stratis-skoulakis-will-be-on-the-use-of-graph-neural-networks-and-reinforcement-learning-for-scheduling-problems-more-specifically-we-will-focus-on-the-most-well-known-industrial-problem-the-job-shop-scheduling-problem",
          title: 'I am happy to share that I will be joining the Laboratory for...',
          description: "",
          section: "News",},{id: "news-i-m-thrilled-to-share-that-i-ve-just-submitted-my-first-paper-to-neurips-it-has-been-an-amazing-and-action-packed-journey-filled-with-both-hard-work-and-enjoyment-now-i-eagerly-await-the-outcome-and-hope-that-my-paper-gets-accepted-stay-tuned-for-updates",
          title: 'I’m thrilled to share that I’ve just submitted my first paper to NeurIPS!...',
          description: "",
          section: "News",},{id: "news-i-m-very-excited-to-announce-that-i-will-be-joining-instadeep-in-paris-for-the-upcoming-six-months-as-a-research-intern-together-with-the-team-i-will-be-working-on-the-development-of-a-general-agent-for-multiple-combinatorial-optimization-problems",
          title: 'I’m very excited to announce that I will be joining InstaDeep in Paris...',
          description: "",
          section: "News",},{id: "news-i-am-happy-to-announce-that-i-will-be-working-at-the-caglar-gulcehre-lab-for-ai-research-claire-on-an-epfl-master-s-research-scholarship-stay-tuned-for-updates-on-the-cool-projects-that-we-re-going-to-take-on",
          title: 'I am happy to announce that I will be working at the Caglar...',
          description: "",
          section: "News",},{id: "projects-algorithms-for-the-3d-bounded-knapsack-problem-with-polyominoes",
          title: 'Algorithms for the 3D bounded knapsack problem with polyominoes',
          description: "This project uses different algorithms to create fast, high-quality solutions to the 3D bin packing problem with one bin",
          section: "Projects",handler: () => {
              window.location.href = "/projects/3d-bin-packing/";
            },},{id: "projects-chronica",
          title: 'Chronica',
          description: "Chronica is a completely on-device Flask-based web application designed to help you create, manage, and visualize notes effectively and securely.",
          section: "Projects",handler: () => {
              window.location.href = "/projects/chronica/";
            },},{id: "projects-context-aware-temporal-modeling-for-anomaly-detection-in-hydropower-systems",
          title: 'Context-Aware Temporal Modeling for Anomaly Detection in Hydropower Systems',
          description: "The final project for CIVIL-426 at EPFL, focusing on developing a context-aware sequence-to-sequence model to detect anomalies in hydropower systems.Sadly, the contents of this project are hidden behind an NDA.",
          section: "Projects",handler: () => {
              window.location.href = "/projects/civil-426/";
            },},{id: "projects-discovering-the-higgs-boson-using-machine-learning",
          title: 'Discovering the Higgs Boson using Machine Learning',
          description: "This project studies different machine learning models applied to the data collected from the experiments performed with the CERN particle accelerator with the aim of discovering the Higgs boson particle",
          section: "Projects",handler: () => {
              window.location.href = "/projects/finding-higgs-boson/";
            },},{id: "projects-emissionaware",
          title: 'EmissionAware',
          description: "For the LauzHack Sustainability 2023 hackathon. Campus sustainability made personal - track your emissions, improve your impact",
          section: "Projects",handler: () => {
              window.location.href = "/projects/lauzhack-sustainability/";
            },},{id: "projects-optimizing-job-allocation-using-reinforcement-learning-with-graph-neural-networks",
          title: 'Optimizing Job Allocation using Reinforcement Learning with Graph Neural Networks',
          description: "This project uses reinforcement learning and graph neural networks to schedule job assignments.",
          section: "Projects",handler: () => {
              window.location.href = "/projects/lions-scheduling/";
            },},{id: "projects-autonomous-lane-changing-using-deep-reinforcement-learning-with-graph-neural-networks",
          title: 'Autonomous Lane Changing using Deep Reinforcement Learning with Graph Neural Networks',
          description: "The aim of this project is to use advanced machine learning methods to solve problems within autonomous driving",
          section: "Projects",handler: () => {
              window.location.href = "/projects/ml4science/";
            },},{id: "projects-probllms-multiple-choice-problem-solving-for-epfl-courses",
          title: 'ProbLLMs: Multiple-Choice Problem Solving for EPFL Courses',
          description: "We develop an AI tutor targeted at STEM education, specifically for multiple-choice question answering related to EPFL courses. Using a general- purpose LLM as a base, we fine-tune a model with enhanced capabilities for complex reasoning tasks related to STEM education.",
          section: "Projects",handler: () => {
              window.location.href = "/projects/mnlp-project/";
            },},{id: "projects-chess-ratings-leveraging-network-methods-to-predict-chess-results",
          title: 'Chess ratings: Leveraging Network Methods to Predict Chess Results',
          description: "We address the problem of estimating chess player ratings and match outcome prediction by leveraging network approaches based on their past games and results.",
          section: "Projects",handler: () => {
              window.location.href = "/projects/network-ml-project/";
            },},{id: "projects-on-the-effect-of-quantization-on-deep-leakage-from-gradients-and-generalization",
          title: 'On the Effect of Quantization on Deep Leakage from Gradients and Generalization',
          description: "We explore various quantization techniques and assess their effectiveness in preserving both data privacy and model performance for machine learning.",
          section: "Projects",handler: () => {
              window.location.href = "/projects/optml-project/";
            },},{id: "projects-strategic-dominion-ai-powered-risk-conquest",
          title: 'Strategic Dominion: AI-Powered Risk Conquest',
          description: "Strategic Dominion is a project where we used the power of machine learning to develop agents for the classic game of Risk, enabling intelligent AI bots to engage in strategic conquests.",
          section: "Projects",handler: () => {
              window.location.href = "/projects/risk-game/";
            },},{id: "projects-the-rl-playground",
          title: 'The RL Playground',
          description: "The goal is to create an interactive book about reinforcement learning",
          section: "Projects",handler: () => {
              window.location.href = "/projects/rl-playground/";
            },},{id: "projects-sustainability-in-high-school-mathematics",
          title: 'Sustainability in High School Mathematics',
          description: "The project aims to integrate sustainability into high school mathematics education by creating a platform that allows students and teachers to easily access and contribute to a library of sustainability-themed exercises and interactive examples.",
          section: "Projects",handler: () => {
              window.location.href = "/projects/sustainability-hs-math/";
            },},{id: "projects-swizz-publication-ready-plots-and-latex-tables-for-ml-papers",
          title: 'Swizz: Publication-ready plots and LaTeX tables for ML papers',
          description: "Swizz is a Python library for generating publication-ready visualizations, LaTeX tables, and subfigure layouts with minimal code and consistent style.",
          section: "Projects",handler: () => {
              window.location.href = "/projects/swizz/";
            },},{id: "projects-tralala-ai-powered-3d-design-platform",
          title: 'Tralala: AI-Powered 3D Design Platform',
          description: "From prompt to prototype to refinement: effortless 3D design with AI. Transform ideas into 3D models using natural language, voice, or images.",
          section: "Projects",handler: () => {
              window.location.href = "/projects/tralala/";
            },},{id: "projects-investigating-predator-prey-dynamics-through-simulated-vision-and-reinforcement-learning",
          title: 'Investigating Predator-Prey Dynamics through Simulated Vision and Reinforcement Learning',
          description: "Course project for the CS-503 Visual Intelligence course at EPFL",
          section: "Projects",handler: () => {
              window.location.href = "/projects/visual-intelligence/";
            },},{id: "projects-wall-m",
          title: 'WALL-M',
          description: "First place @ HackUPC 2024Bringing structure to your e-mail inbox filled with personalized unstructured data and helping you communicate with it using RAG and hybrid search with minimal hallucinations.",
          section: "Projects",handler: () => {
              window.location.href = "/projects/wall-m/";
            },},{
        id: 'social-email',
        title: 'email',
        section: 'Socials',
        handler: () => {
          window.open("mailto:%6C%61%72%73%71%75%61%65%64%76%6C%69%65%67@%6F%75%74%6C%6F%6F%6B.%63%6F%6D", "_blank");
        },
      },{
        id: 'social-github',
        title: 'GitHub',
        section: 'Socials',
        handler: () => {
          window.open("https://github.com/lars-quaedvlieg", "_blank");
        },
      },{
        id: 'social-linkedin',
        title: 'LinkedIn',
        section: 'Socials',
        handler: () => {
          window.open("https://www.linkedin.com/in/lars-quaedvlieg", "_blank");
        },
      },{
        id: 'social-orcid',
        title: 'ORCID',
        section: 'Socials',
        handler: () => {
          window.open("https://orcid.org/0000-0002-0109-5705", "_blank");
        },
      },{
        id: 'social-scholar',
        title: 'Google Scholar',
        section: 'Socials',
        handler: () => {
          window.open("https://scholar.google.com/citations?user=f_-rgVcAAAAJ", "_blank");
        },
      },{
        id: 'social-x',
        title: 'X',
        section: 'Socials',
        handler: () => {
          window.open("https://twitter.com/lars_quaedvlieg", "_blank");
        },
      },];
