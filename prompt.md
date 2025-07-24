<problem-description>
You are a highly-skilled professor in AI planning generating a plan for a PDDL task from the domain <domain>$name</domain>. You will be given the PDDL domain and the PDDL task, and you need to return the plan between the tags <plan> and </plan>. You will receive a two examples to help you in generating the plan.
</problem-description> 

<problem-description-with-landmarks>
You are a highly-skilled professor in AI planning generating a plan for a PDDL task from the domain <domain>$name</domain>. You will be given the PDDL domain, the PDDL task, and a set of action landmarks of the task. Action landmarks are actions that must be part of any valid plan for the task. You need to return the plan between the tags <plan> and </plan>. You will receive a two examples to help you in generating the plan.
</problem-description-with-landmarks> 

This is the PDDL domain file of the $name domain:
<domain-file>
$domain
</domain-file>

This is the PDDL task file, for which you need to generate a plan:
<task-file>
$instance
</task-file>

This is a set of action landmarks for the task you need to generate a plan for:
<landmarks-set>
$landmarks
</landmarks-set>

This is the PDDL domain file of another domain, called Storage, which serves as an example:
<storage-domain-file-example>
</storage-domain-file-example>

This is an example of an task file from the Storage domain:
<storage-task-file-example>
</storage-task-file-example>

This is a set of action landmarks for the Storage task above:
<storage-landmarks-set-example>
</storage-landmarks-set-example>

This is a plan for the Storage task above:
<plan-storage-example>
</plan-storage-example>

This is the PDDL domain file of another domain, called Hanoi, which serves as an example:
<hanoi-domain-file-example>
</hanoi-domain-file-example>

This is an example of an task file from the Hanoi domain:
<hanoi-task-file-example>
</hanoi-task-file-example>

This is a set of action landmarks for the Hanoi task above:
<hanoi-landmarks-set-example>
</hanoi-landmarks-set-example>

This is a plan for the Hanoi task above:
<plan-hanoi-example>
</plan-hanoi-example>

Provide only the plan for the given task. Here is a checklist to help you with your problem:
<checklist>
1) The plan must be in the same format as the examples above.
2) The plan should be preceded by the <plan> tag and should be followed by the </plan> tag.
3) The actions in the plan must be from the set of actions in the domain described above, that is, they must use the same name and the same number of parameters as one of the action schemas.
4) The plan must be valid, that is, each action must be applicable in the state it is applied, and the plan must end in a goal state.
</checklist>