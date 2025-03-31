; child-snack task with 1 children and 0.49 gluten factor 
; constant factor of 1.7
; random seed: 606284

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child127 - child
    bread180 - bread-portion
    content347 - content-portion
    tray370 tray430 - tray
    table36 table9 table461 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray370 kitchen)
     (at tray430 kitchen)
     (at_kitchen_bread bread180)
     (at_kitchen_content content347)
     (not_allergic_gluten child127)
     (waiting child127 table9)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child127)
    )
  )
)
