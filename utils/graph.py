import streamlit as st

def similarity_graph_output():
    graph = st.empty()
    st.session_state.graph_component = graph

# def final_graph():
#     final_graph_bt = st.button('View Full Graphs')
#     if final_graph_bt:
#         fig = px.line(pd.DataFrame(st.session_state.prompt_score_dict), title='Similarity Graph')
#         st.plotly_chart(fig,use_container_width=True)
