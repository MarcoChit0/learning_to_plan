; child-snack task with 2 children and 0.54 gluten factor 
; constant factor of 1.7
; random seed: 228600

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child415 child258 - child
    bread449 bread349 - bread-portion
    content133 content495 - content-portion
    tray388 - tray
    table318 table319 table130 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray388 kitchen)
     (at_kitchen_bread bread449)
     (at_kitchen_bread bread349)
     (at_kitchen_content content133)
     (at_kitchen_content content495)
     (no_gluten_bread bread449)
     (no_gluten_content content133)
     (allergic_gluten child258)
     (not_allergic_gluten child415)
     (waiting child415 table318)
     (waiting child258 table318)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child415)
     (served child258)
    )
  )
)
