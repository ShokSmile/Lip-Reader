# Project: Lip movements Recognition and Virtual Conversationalist

Group: Tarasov Aleksandr, Sellage Kulakshi Bhashini Fernando (p24)


## Description

---

Our project began with the idea of implementing a system for speech recognition based on lip movement. However, after conducting an extensive review of the field, we discovered a unique end-to-end solution provided by the META team. This solution, available at [auto_avsr](https://github.com/mpc001/auto_avsr), not only enables speech recognition based on lip movement but also allows integration of an audio module (though we did not utilize this feature in our project).

Upon closer examination, we concluded that implementing this solution in its original form would be too straightforward. As a result, we decided to expand the project by creating a web application that incorporates three main models.

1. **Lip Movement Speech Recognition Model:**
   - We integrated the model from META, provided in the [auto_avsr](https://github.com/mpc001/auto_avsr) repository. This model employs an advanced algorithm for speech recognition based on lip movement.

2. **Generative Text Response Model:**
   - To create a virtual conversationalist capable of responding to phrases, we introduced a generative model that generates textual responses to input queries.

3. **Generative Voice Synthesis Model:**
   - In addition to the text response, we developed a second generative model that transforms the generated text into realistic voice synthesis.

We chose the Streamlit framework for developing the web application, providing users with a user-friendly interface for accessing our machine learning models directly through a web browser. Thus, our online ML application offers users the opportunity to interact with cutting-edge speech recognition and generation technologies.


## How to use it

---


