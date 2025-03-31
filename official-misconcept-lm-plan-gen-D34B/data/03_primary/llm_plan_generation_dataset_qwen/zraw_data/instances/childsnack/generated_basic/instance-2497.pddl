; child-snack task with 2 children and 0.39 gluten factor 
; constant factor of 1.9
; random seed: 360975

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child332 child274 - child
    bread466 bread247 - bread-portion
    content337 content2 - content-portion
    tray24 tray253 - tray
    table29 table115 table33 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray24 kitchen)
     (at tray253 kitchen)
     (at_kitchen_bread bread466)
     (at_kitchen_bread bread247)
     (at_kitchen_content content337)
     (at_kitchen_content content2)
     (not_allergic_gluten child274)
     (not_allergic_gluten child332)
     (waiting child332 table29)
     (waiting child274 table115)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child332)
     (served child274)
    )
  )
)
