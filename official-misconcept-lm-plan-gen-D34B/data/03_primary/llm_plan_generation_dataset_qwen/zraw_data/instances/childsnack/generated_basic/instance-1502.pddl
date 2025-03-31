; child-snack task with 2 children and 0.66 gluten factor 
; constant factor of 1.3
; random seed: 82841

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child272 child157 - child
    bread211 bread69 - bread-portion
    content241 content310 - content-portion
    tray463 tray357 tray492 - tray
    table482 table309 table490 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray463 kitchen)
     (at tray357 kitchen)
     (at tray492 kitchen)
     (at_kitchen_bread bread211)
     (at_kitchen_bread bread69)
     (at_kitchen_content content241)
     (at_kitchen_content content310)
     (no_gluten_bread bread211)
     (no_gluten_content content241)
     (allergic_gluten child272)
     (not_allergic_gluten child157)
     (waiting child272 table490)
     (waiting child157 table482)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child272)
     (served child157)
    )
  )
)
