; child-snack task with 2 children and 0.55 gluten factor 
; constant factor of 1.7
; random seed: 22431

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child157 child211 - child
    bread71 bread468 - bread-portion
    content429 content303 - content-portion
    tray466 - tray
    table328 table237 table384 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray466 kitchen)
     (at_kitchen_bread bread71)
     (at_kitchen_bread bread468)
     (at_kitchen_content content429)
     (at_kitchen_content content303)
     (no_gluten_bread bread71)
     (no_gluten_content content303)
     (allergic_gluten child157)
     (not_allergic_gluten child211)
     (waiting child157 table328)
     (waiting child211 table237)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child157)
     (served child211)
    )
  )
)
