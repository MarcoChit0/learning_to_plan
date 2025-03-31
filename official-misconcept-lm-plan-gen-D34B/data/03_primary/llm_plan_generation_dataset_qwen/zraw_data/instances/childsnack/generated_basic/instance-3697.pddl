; child-snack task with 2 children and 0.8 gluten factor 
; constant factor of 1.7
; random seed: 136438

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child252 child96 - child
    bread237 bread432 - bread-portion
    content119 content201 - content-portion
    tray280 tray450 - tray
    table461 table95 table223 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray280 kitchen)
     (at tray450 kitchen)
     (at_kitchen_bread bread237)
     (at_kitchen_bread bread432)
     (at_kitchen_content content119)
     (at_kitchen_content content201)
     (no_gluten_bread bread237)
     (no_gluten_content content119)
     (allergic_gluten child252)
     (not_allergic_gluten child96)
     (waiting child252 table461)
     (waiting child96 table461)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child252)
     (served child96)
    )
  )
)
