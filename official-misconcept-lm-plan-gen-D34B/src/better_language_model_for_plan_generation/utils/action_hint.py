def obtain_action_hint_text(plan_text, action_hint_size, is_llama_factory, anchor_plan_sec):
    output = ""
    if is_llama_factory: # llama_factory does not contain the anchor_plan_sec, thus we need to add it
        output += anchor_plan_sec
        
    # plan_text also contain the anchor_plan_sec
    plan_text_start_idx = plan_text.find(anchor_plan_sec)
    if plan_text_start_idx == -1:
        raise ValueError("anchor_plan_sec not found in plan_text")
    plan_text = plan_text[plan_text_start_idx + len(anchor_plan_sec):]
    plan_text = plan_text.strip()
    # split the plan_text into a list of actions
    plan_text_lst = plan_text.split("\n")
    # remove the empty string
    plan_text_lst = [x.strip() for x in plan_text_lst if x != ""]
    # also remove ones contain <BACK>
    plan_text_lst = [x for x in plan_text_lst if "WRONG" not in x]
    # get the first action_hint_size actions
    action_hint_lst = plan_text_lst[:action_hint_size]
    # * special case, add space before < and after >
    for idx in range(len(action_hint_lst)):
        if action_hint_lst[idx].startswith("<"):
            action_hint_lst[idx] = " " + action_hint_lst[idx]
        if action_hint_lst[idx].endswith(">"):
            action_hint_lst[idx] = action_hint_lst[idx] + " "
    # join the action_hint_lst
    output += "\n".join(action_hint_lst)
    # add a new line at the end
    output += "\n"
    return output