; child-snack task with 2 children and 0.53 gluten factor 
; constant factor of 1.9
; random seed: 526642

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child329 child416 - child
    bread4 bread147 - bread-portion
    content10 content280 - content-portion
    tray12 tray438 tray90 - tray
    table246 table26 table223 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray12 kitchen)
     (at tray438 kitchen)
     (at tray90 kitchen)
     (at_kitchen_bread bread4)
     (at_kitchen_bread bread147)
     (at_kitchen_content content10)
     (at_kitchen_content content280)
     (no_gluten_bread bread4)
     (no_gluten_content content10)
     (allergic_gluten child329)
     (not_allergic_gluten child416)
     (waiting child329 table26)
     (waiting child416 table223)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child329)
     (served child416)
    )
  )
)
