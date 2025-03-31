; child-snack task with 2 children and 0.77 gluten factor 
; constant factor of 1.3
; random seed: 498237

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child176 child454 - child
    bread168 bread305 - bread-portion
    content262 content133 - content-portion
    tray89 - tray
    table375 table389 table109 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray89 kitchen)
     (at_kitchen_bread bread168)
     (at_kitchen_bread bread305)
     (at_kitchen_content content262)
     (at_kitchen_content content133)
     (no_gluten_bread bread305)
     (no_gluten_content content262)
     (allergic_gluten child454)
     (not_allergic_gluten child176)
     (waiting child176 table375)
     (waiting child454 table389)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child176)
     (served child454)
    )
  )
)
