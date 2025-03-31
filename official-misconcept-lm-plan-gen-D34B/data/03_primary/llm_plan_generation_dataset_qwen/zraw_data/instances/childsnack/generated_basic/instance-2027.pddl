; child-snack task with 2 children and 0.34 gluten factor 
; constant factor of 1.5
; random seed: 379737

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child109 child139 - child
    bread450 bread497 - bread-portion
    content281 content370 - content-portion
    tray117 tray419 tray454 - tray
    table196 table248 table15 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray117 kitchen)
     (at tray419 kitchen)
     (at tray454 kitchen)
     (at_kitchen_bread bread450)
     (at_kitchen_bread bread497)
     (at_kitchen_content content281)
     (at_kitchen_content content370)
     (not_allergic_gluten child109)
     (not_allergic_gluten child139)
     (waiting child109 table196)
     (waiting child139 table15)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child109)
     (served child139)
    )
  )
)
