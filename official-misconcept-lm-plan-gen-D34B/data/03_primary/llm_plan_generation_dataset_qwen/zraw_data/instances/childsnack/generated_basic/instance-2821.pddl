; child-snack task with 2 children and 0.32 gluten factor 
; constant factor of 1.7
; random seed: 593755

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child374 child268 - child
    bread149 bread302 - bread-portion
    content497 content433 - content-portion
    tray195 tray123 - tray
    table453 table126 table305 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray195 kitchen)
     (at tray123 kitchen)
     (at_kitchen_bread bread149)
     (at_kitchen_bread bread302)
     (at_kitchen_content content497)
     (at_kitchen_content content433)
     (not_allergic_gluten child374)
     (not_allergic_gluten child268)
     (waiting child374 table126)
     (waiting child268 table126)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child374)
     (served child268)
    )
  )
)
