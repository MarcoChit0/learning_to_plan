; child-snack task with 2 children and 0.58 gluten factor 
; constant factor of 1.5
; random seed: 499718

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child367 child47 - child
    bread120 bread83 - bread-portion
    content128 content348 - content-portion
    tray474 tray482 tray112 - tray
    table433 table462 table411 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray474 kitchen)
     (at tray482 kitchen)
     (at tray112 kitchen)
     (at_kitchen_bread bread120)
     (at_kitchen_bread bread83)
     (at_kitchen_content content128)
     (at_kitchen_content content348)
     (no_gluten_bread bread120)
     (no_gluten_content content348)
     (allergic_gluten child367)
     (not_allergic_gluten child47)
     (waiting child367 table433)
     (waiting child47 table433)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child367)
     (served child47)
    )
  )
)
