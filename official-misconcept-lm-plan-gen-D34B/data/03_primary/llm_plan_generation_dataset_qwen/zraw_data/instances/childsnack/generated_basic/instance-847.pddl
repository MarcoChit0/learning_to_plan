; child-snack task with 2 children and 0.59 gluten factor 
; constant factor of 1.9
; random seed: 798763

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child480 child364 - child
    bread179 bread117 - bread-portion
    content65 content313 - content-portion
    tray46 tray453 - tray
    table354 table164 table202 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray46 kitchen)
     (at tray453 kitchen)
     (at_kitchen_bread bread179)
     (at_kitchen_bread bread117)
     (at_kitchen_content content65)
     (at_kitchen_content content313)
     (no_gluten_bread bread117)
     (no_gluten_content content65)
     (allergic_gluten child364)
     (not_allergic_gluten child480)
     (waiting child480 table354)
     (waiting child364 table354)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child480)
     (served child364)
    )
  )
)
