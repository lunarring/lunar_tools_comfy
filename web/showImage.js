import { app } from "../../../scripts/app.js";

app.registerExtension({
    name: "LR.Show_Image",
    async beforeRegisterNodeDef(n, t, e) {
        if (t.name === "LR_Show_Image") {
            //console.log("Registering LR_Show_Image node...");

            function showImage(imageData) {
                // console.log("showImage called with data:", imageData);

                // Convert imageData to string if it is an array
                if (Array.isArray(imageData)) {
                    // console.log("Converting imageData array to string.");
                    imageData = imageData.join("");
                }

                // Create base64 image source
                const imgSrc = `data:image/png;base64,${imageData}`;

                // First execution: create the image container and img element
                if (!this.imageContainer) {
                    // console.log("Creating image container with addDOMWidget.");

                    // Create div container for the image
                    const container = document.createElement("div");
                    container.style.padding = "5px";
                    container.style.width = "100%";
                    container.style.textAlign = "center";

                    // Create img element and set initial src
                    const img = document.createElement("img");
                    img.src = imgSrc;
                    img.style.maxWidth = "100%";
                    img.style.borderRadius = "5px";

                    // Store img element reference for future updates
                    this.imageContainer = img;
                    container.appendChild(img);

                    // Add container as a widget
                    this.addDOMWidget("Image Display", "custom", container);

                    // Adjust node size after the image loads
                    img.onload = () => setSizeForImage.call(this);
                } else {
                    // Subsequent executions: update the img src directly
                    // console.log("Updating image src in existing image container.");
                    this.imageContainer.src = imgSrc;
                    setSizeForImage.call(this);
                }
                // console.log("Image element added to the node.");
            }

            // Function to set the node size according to the image dimensions
            function setSizeForImage() {
                const img = this.imageContainer;
                const padding = 20; // Adjust padding if needed
                const newWidth = img.naturalWidth + padding;
                const newHeight = img.naturalHeight + padding;

                // Set the node's size based on the image dimensions
                this.setSize([newWidth, newHeight]);
                // console.log(`Node resized to: ${newWidth}x${newHeight}`);
            }

            // Override onExecuted and onConfigure functions to apply showImage
            const originalOnExecuted = n.prototype.onExecuted;
            n.prototype.onExecuted = function (data) {
                // console.log("onExecuted called with data:", data);
                originalOnExecuted?.apply(this, arguments);
                showImage.call(this, data.image);
            };

            const originalOnConfigure = n.prototype.onConfigure;
            n.prototype.onConfigure = function () {
                // console.log("onConfigure called with widgets_values:", this.widgets_values);
                originalOnConfigure?.apply(this, arguments);
                if (this.widgets_values?.length) {
                    showImage.call(this, this.widgets_values[0]);
                }
            };
        }
    }
});
