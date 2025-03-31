; child-snack task with 1 children and 0.36 gluten factor 
; constant factor of 1.5
; random seed: 177680

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child408 - child
    bread477 - bread-portion
    content484 - content-portion
    tray129 - tray
    table195 table130 table234 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray129 kitchen)
     (at_kitchen_bread bread477)
     (at_kitchen_content content484)
     (not_allergic_gluten child408)
     (waiting child408 table195)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child408)
    )
  )
)
