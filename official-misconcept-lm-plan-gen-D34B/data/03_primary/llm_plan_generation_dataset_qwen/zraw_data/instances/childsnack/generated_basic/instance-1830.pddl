; child-snack task with 2 children and 0.58 gluten factor 
; constant factor of 1.9
; random seed: 583150

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child335 child203 - child
    bread382 bread171 - bread-portion
    content467 content412 - content-portion
    tray35 - tray
    table332 table299 table10 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray35 kitchen)
     (at_kitchen_bread bread382)
     (at_kitchen_bread bread171)
     (at_kitchen_content content467)
     (at_kitchen_content content412)
     (no_gluten_bread bread382)
     (no_gluten_content content467)
     (allergic_gluten child203)
     (not_allergic_gluten child335)
     (waiting child335 table10)
     (waiting child203 table332)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child335)
     (served child203)
    )
  )
)
