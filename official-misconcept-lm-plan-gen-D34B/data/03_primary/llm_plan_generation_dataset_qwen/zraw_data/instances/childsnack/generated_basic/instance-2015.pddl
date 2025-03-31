; child-snack task with 2 children and 0.34 gluten factor 
; constant factor of 1.5
; random seed: 185710

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child242 child421 - child
    bread424 bread460 - bread-portion
    content239 content257 - content-portion
    tray82 tray258 tray388 - tray
    table478 table39 table296 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray82 kitchen)
     (at tray258 kitchen)
     (at tray388 kitchen)
     (at_kitchen_bread bread424)
     (at_kitchen_bread bread460)
     (at_kitchen_content content239)
     (at_kitchen_content content257)
     (not_allergic_gluten child242)
     (not_allergic_gluten child421)
     (waiting child242 table39)
     (waiting child421 table296)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child242)
     (served child421)
    )
  )
)
