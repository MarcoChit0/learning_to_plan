; child-snack task with 2 children and 0.64 gluten factor 
; constant factor of 1.3
; random seed: 311065

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child14 child228 - child
    bread411 bread176 - bread-portion
    content88 content237 - content-portion
    tray196 - tray
    table40 table228 table34 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray196 kitchen)
     (at_kitchen_bread bread411)
     (at_kitchen_bread bread176)
     (at_kitchen_content content88)
     (at_kitchen_content content237)
     (no_gluten_bread bread411)
     (no_gluten_content content237)
     (allergic_gluten child228)
     (not_allergic_gluten child14)
     (waiting child14 table228)
     (waiting child228 table40)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child14)
     (served child228)
    )
  )
)
