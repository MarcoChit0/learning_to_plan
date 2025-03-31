; child-snack task with 2 children and 0.64 gluten factor 
; constant factor of 1.3
; random seed: 46659

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child307 child314 - child
    bread87 bread198 - bread-portion
    content491 content333 - content-portion
    tray207 - tray
    table57 table260 table474 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray207 kitchen)
     (at_kitchen_bread bread87)
     (at_kitchen_bread bread198)
     (at_kitchen_content content491)
     (at_kitchen_content content333)
     (no_gluten_bread bread87)
     (no_gluten_content content333)
     (allergic_gluten child314)
     (not_allergic_gluten child307)
     (waiting child307 table57)
     (waiting child314 table57)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child307)
     (served child314)
    )
  )
)
