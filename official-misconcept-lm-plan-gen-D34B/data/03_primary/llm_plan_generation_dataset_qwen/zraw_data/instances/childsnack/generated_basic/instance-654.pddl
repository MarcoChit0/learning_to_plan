; child-snack task with 2 children and 0.63 gluten factor 
; constant factor of 1.3
; random seed: 746898

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child400 child454 - child
    bread218 bread87 - bread-portion
    content486 content390 - content-portion
    tray300 tray66 - tray
    table202 table319 table466 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray300 kitchen)
     (at tray66 kitchen)
     (at_kitchen_bread bread218)
     (at_kitchen_bread bread87)
     (at_kitchen_content content486)
     (at_kitchen_content content390)
     (no_gluten_bread bread87)
     (no_gluten_content content486)
     (allergic_gluten child400)
     (not_allergic_gluten child454)
     (waiting child400 table319)
     (waiting child454 table466)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child400)
     (served child454)
    )
  )
)
