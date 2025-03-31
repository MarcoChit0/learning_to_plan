; child-snack task with 1 children and 0.23 gluten factor 
; constant factor of 1.7
; random seed: 126159

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child29 - child
    bread59 - bread-portion
    content204 - content-portion
    tray60 tray42 - tray
    table390 table409 table295 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray60 kitchen)
     (at tray42 kitchen)
     (at_kitchen_bread bread59)
     (at_kitchen_content content204)
     (not_allergic_gluten child29)
     (waiting child29 table409)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child29)
    )
  )
)
