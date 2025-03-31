; child-snack task with 2 children and 0.64 gluten factor 
; constant factor of 1.9
; random seed: 278116

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child378 child385 - child
    bread281 bread306 - bread-portion
    content200 content164 - content-portion
    tray450 tray84 - tray
    table179 table297 table494 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray450 kitchen)
     (at tray84 kitchen)
     (at_kitchen_bread bread281)
     (at_kitchen_bread bread306)
     (at_kitchen_content content200)
     (at_kitchen_content content164)
     (no_gluten_bread bread306)
     (no_gluten_content content200)
     (allergic_gluten child378)
     (not_allergic_gluten child385)
     (waiting child378 table297)
     (waiting child385 table494)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child378)
     (served child385)
    )
  )
)
