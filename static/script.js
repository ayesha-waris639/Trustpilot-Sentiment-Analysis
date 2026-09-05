const reviewInput =
    document.getElementById("review");

const charCount =
    document.getElementById("charCount");


/* Character counter */

reviewInput.addEventListener(
    "input",
    function () {

        charCount.textContent =
            this.value.length;

    }
);


/* Example reviews */

const examples = {

    positive:
        "I absolutely love this service! Everything was excellent and the delivery was incredibly fast.",

    negative:
        "This was a terrible experience. The service was disappointing and I will not use it again.",

    neutral:
        "The package arrived today. The delivery took three days."
};


function useExample(type) {

    reviewInput.value =
        examples[type];

    charCount.textContent =
        reviewInput.value.length;

    reviewInput.focus();
}


/* Analyze */

async function analyzeSentiment() {

    const text =
        reviewInput.value.trim();

    const resultCard =
        document.getElementById(
            "resultCard"
        );

    const loading =
        document.getElementById(
            "loading"
        );

    const button =
        document.getElementById(
            "analyzeBtn"
        );


    if (!text) {

        alert(
            "Please enter a review first."
        );

        reviewInput.focus();

        return;
    }


    loading.classList.remove(
        "hidden"
    );

    resultCard.classList.add(
        "hidden"
    );

    button.disabled = true;


    try {

        const response =
            await fetch(
                "/predict",
                {

                    method: "POST",

                    headers: {
                        "Content-Type":
                            "application/json"
                    },

                    body: JSON.stringify({
                        text: text
                    })

                }
            );


        const data =
            await response.json();


        if (!response.ok ||
            !data.success) {

            throw new Error(
                data.error ||
                "Prediction failed."
            );
        }


        /* Sentiment */

        const sentimentBadge =
            document.getElementById(
                "sentimentBadge"
            );

        sentimentBadge.textContent =
            data.sentiment;


        sentimentBadge.style.background =
            getSentimentBackground(
                data.sentiment
            );

        sentimentBadge.style.color =
            getSentimentColor(
                data.sentiment
            );


        /* Confidence */

        document.getElementById(
            "confidence"
        ).textContent =
            data.confidence + "%";


        document.getElementById(
            "confidenceBar"
        ).style.width =
            data.confidence + "%";


        /* Probabilities */

        document.getElementById(
            "negative"
        ).textContent =
            data.probabilities.Negative
            + "%";


        document.getElementById(
            "neutral"
        ).textContent =
            data.probabilities.Neutral
            + "%";


        document.getElementById(
            "positive"
        ).textContent =
            data.probabilities.Positive
            + "%";


        /* Review */

        document.getElementById(
            "reviewPreview"
        ).textContent =
            text;


        resultCard.classList.remove(
            "hidden"
        );


        resultCard.scrollIntoView({
            behavior: "smooth",
            block: "start"
        });

    }

    catch (error) {

        alert(
            "Error: " +
            error.message
        );

    }

    finally {

        loading.classList.add(
            "hidden"
        );

        button.disabled =
            false;
    }
}


/* Sentiment colors */

function getSentimentBackground(
    sentiment
) {

    if (
        sentiment === "Negative"
    ) {

        return "#fef2f2";

    }

    if (
        sentiment === "Neutral"
    ) {

        return "#fffbeb";

    }

    return "#ecfdf5";
}


function getSentimentColor(
    sentiment
) {

    if (
        sentiment === "Negative"
    ) {

        return "#dc2626";

    }

    if (
        sentiment === "Neutral"
    ) {

        return "#d97706";

    }

    return "#059669";
}


/* Clear */

function clearText() {

    reviewInput.value = "";

    charCount.textContent = "0";

    document
        .getElementById("resultCard")
        .classList.add("hidden");

    reviewInput.focus();
}


/* Ctrl + Enter */

reviewInput.addEventListener(
    "keydown",
    function (event) {

        if (
            event.ctrlKey &&
            event.key === "Enter"
        ) {

            analyzeSentiment();

        }

    }
);