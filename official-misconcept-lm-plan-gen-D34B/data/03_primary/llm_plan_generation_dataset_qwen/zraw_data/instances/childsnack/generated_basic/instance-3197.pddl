; child-snack task with 2 children and 0.75 gluten factor 
; constant factor of 1.7
; random seed: 161929

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child17 child22 - child
    bread327 bread381 - bread-portion
    content216 content353 - content-portion
    tray181 tray472 - tray
    table150 table77 table120 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray181 kitchen)
     (at tray472 kitchen)
     (at_kitchen_bread bread327)
     (at_kitchen_bread bread381)
     (at_kitchen_content content216)
     (at_kitchen_content content353)
     (no_gluten_bread bread381)
     (no_gluten_content content216)
     (allergic_gluten child17)
     (not_allergic_gluten child22)
     (waiting child17 table77)
     (waiting child22 table120)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child17)
     (served child22)
    )
  )
)
