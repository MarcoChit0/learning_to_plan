; child-snack task with 2 children and 0.75 gluten factor 
; constant factor of 1.7
; random seed: 55374

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child19 child455 - child
    bread284 bread16 - bread-portion
    content213 content309 - content-portion
    tray80 tray18 - tray
    table10 table280 table67 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray80 kitchen)
     (at tray18 kitchen)
     (at_kitchen_bread bread284)
     (at_kitchen_bread bread16)
     (at_kitchen_content content213)
     (at_kitchen_content content309)
     (no_gluten_bread bread16)
     (no_gluten_content content309)
     (allergic_gluten child19)
     (not_allergic_gluten child455)
     (waiting child19 table10)
     (waiting child455 table280)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child19)
     (served child455)
    )
  )
)
