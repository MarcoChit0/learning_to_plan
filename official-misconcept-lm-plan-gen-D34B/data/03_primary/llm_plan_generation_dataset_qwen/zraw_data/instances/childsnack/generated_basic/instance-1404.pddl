; child-snack task with 2 children and 0.55 gluten factor 
; constant factor of 1.7
; random seed: 726047

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child153 child119 - child
    bread354 bread292 - bread-portion
    content169 content439 - content-portion
    tray208 - tray
    table120 table470 table397 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray208 kitchen)
     (at_kitchen_bread bread354)
     (at_kitchen_bread bread292)
     (at_kitchen_content content169)
     (at_kitchen_content content439)
     (no_gluten_bread bread354)
     (no_gluten_content content169)
     (allergic_gluten child153)
     (not_allergic_gluten child119)
     (waiting child153 table120)
     (waiting child119 table120)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child153)
     (served child119)
    )
  )
)
