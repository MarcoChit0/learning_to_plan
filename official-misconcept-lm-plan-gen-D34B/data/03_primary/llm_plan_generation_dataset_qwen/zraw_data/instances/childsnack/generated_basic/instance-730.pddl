; child-snack task with 2 children and 0.64 gluten factor 
; constant factor of 1.9
; random seed: 484315

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child225 child351 - child
    bread397 bread0 - bread-portion
    content172 content71 - content-portion
    tray123 tray227 - tray
    table130 table214 table312 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray123 kitchen)
     (at tray227 kitchen)
     (at_kitchen_bread bread397)
     (at_kitchen_bread bread0)
     (at_kitchen_content content172)
     (at_kitchen_content content71)
     (no_gluten_bread bread397)
     (no_gluten_content content71)
     (allergic_gluten child225)
     (not_allergic_gluten child351)
     (waiting child225 table214)
     (waiting child351 table312)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child225)
     (served child351)
    )
  )
)
