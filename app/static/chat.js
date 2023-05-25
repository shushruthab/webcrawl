class Chatbot {
    constructor() {
        this.args = {
            openBot: document.querySelector('.chatbot-button'),
            chatBot: document.querySelector('.chatbot-support'),
            sendButton: document.querySelector('.send-button')
        }
        console.log(this.args);
        this.state = false;
        this.messages = [];
    }

    display() {
        const {openBot, chatBot, sendButton} = this.args;

        openBot.addEventListener('click', () => this.toggleState(chatBot))

        sendButton.addEventListener('click', () => this.onSendButton(chatBot))

        const node = chatBot.querySelector('input');
        node.addEventListener('keyup', ({key}) => {
            if (key === 'Enter') {
                this.onSendButton(chatBot)
            }
        })
    }

    toggleState(chatBot) {
        console.log('Toggle state called');
        this.state = !this.state;

        // show or hide the chat window
        if (this.state) {
            chatBot.classList.add('chatbot--active');
        } else {
            chatBot.classList.remove('chatbot--active');
        }
    }

    onSendButton(chatBot) {
        let textField = chatBot.querySelector('input');
        let text1 = textField.value;
        if (text1 === "") {
            return;
        }

        let msg1 = {name: "User", message: text1}
        this.messages.push(msg1);
        this.updateChatText(chatBot);

        let typingMsg = {name: "Missy", message: "<i>typing...</i>"};
        this.messages.push(typingMsg);
        this.updateChatText(chatBot); 

        fetch('/chat', {
            method: 'POST',
            body: JSON.stringify({message: text1}),
            mode: 'cors',
            headers: {
                'Content-Type': 'application/json'
            },

        })
        .then(r => r.json())
        .then(r => {
            this.messages.pop();
            let msg2 = {name: "Missy", message: r.answer};
            this.messages.push(msg2);
            this.updateChatText(chatBot);
            textField.value = '';
    }).catch((error) => {
        console.error('Error:', error);
        this.updateChatText(chatBot);
        textField.value = '';
    })
}
    updateChatText(chatBot) {
        var html = '';
        this.messages.slice().reverse().forEach(function(item) {
            if (item.name === "Missy") 
            {
                html += '<div class="messages-item messages-item--visitor">' + item.message + '</div>'
            } else {
                html += '<div class="messages-item messages-item--operator">' + item.message + '</div>'
            }
        });
        const chatmessage = chatBot.querySelector('.chatbot-messages');
        chatmessage.innerHTML = html;
    }
}

const chatbot = new Chatbot();
chatbot.display();