; child-snack task with 3 children and 0.33 gluten factor 
; constant factor of 1.7
; random seed: 960899

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child48 child240 child361 - child
    bread490 bread403 bread67 - bread-portion
    content488 content448 content171 - content-portion
    tray48 tray349 tray54 tray287 - tray
    table478 table341 table58 - place
    sandw1 sandw2 sandw3 sandw4 sandw5 sandw6 - sandwich
  )
  (:init
     (at tray48 kitchen)
     (at tray349 kitchen)
     (at tray54 kitchen)
     (at tray287 kitchen)
     (at_kitchen_bread bread490)
     (at_kitchen_bread bread403)
     (at_kitchen_bread bread67)
     (at_kitchen_content content488)
     (at_kitchen_content content448)
     (at_kitchen_content content171)
     (not_allergic_gluten child48)
     (not_allergic_gluten child361)
     (not_allergic_gluten child240)
     (waiting child48 table478)
     (waiting child240 table478)
     (waiting child361 table478)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
     (notexist sandw5)
     (notexist sandw6)
  )
  (:goal
    (and
     (served child48)
     (served child240)
     (served child361)
    )
  )
)
