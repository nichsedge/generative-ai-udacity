In this exercise, we'll use Memory within a Chain to keep track of a conversation summary. This is another widely used pattern to work around limitations of a context window. 

At the same time, we'll personalize movie ratings based on a set of personal question / answer pairs. 

The overall recommendation algorithm is as follows:
- pick a list of movies you'd like to rate. 
- create a list of q/a pairs you think  would be helpful for LLM in recommending a movie for you. 
- simulate a conversation between you and LLM to create a context for your movie preferences based on your q/a pairs
- then, for every movie you'd like to rate:
    - fetch plot summary of a movie from wikipedia (code for that is provided)
    - Ask LLM to rate the movie based on the plot summary and your preferences
    - Use ConversationSummaryMemory to keep track of the conversation summary and latest best rating without adding all the movie plots into the context. This helps keep the overall context length small to fit into LLM context window, and also keeps the cost of LLM calls lower. 
- finally, when all the movies are rated, ask LLM to recommend one movie that human will ask the most. At this point, you've been keeping the context updated with the movie ratings, and it should give an LLM enough information to do the final recommendation. 

After this exercise, you'll have a much better hands on experience with using Chains to solve real world problems.  



