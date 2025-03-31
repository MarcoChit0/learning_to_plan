; child-snack task with 2 children and 0.78 gluten factor 
; constant factor of 1.9
; random seed: 754582

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child337 child332 - child
    bread143 bread126 - bread-portion
    content429 content476 - content-portion
    tray233 - tray
    table429 table431 table206 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray233 kitchen)
     (at_kitchen_bread bread143)
     (at_kitchen_bread bread126)
     (at_kitchen_content content429)
     (at_kitchen_content content476)
     (no_gluten_bread bread126)
     (no_gluten_content content476)
     (allergic_gluten child332)
     (not_allergic_gluten child337)
     (waiting child337 table206)
     (waiting child332 table429)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child337)
     (served child332)
    )
  )
)
