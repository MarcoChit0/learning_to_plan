; child-snack task with 2 children and 0.63 gluten factor 
; constant factor of 1.7
; random seed: 313986

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child238 child54 - child
    bread275 bread21 - bread-portion
    content280 content194 - content-portion
    tray370 tray242 - tray
    table321 table466 table249 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray370 kitchen)
     (at tray242 kitchen)
     (at_kitchen_bread bread275)
     (at_kitchen_bread bread21)
     (at_kitchen_content content280)
     (at_kitchen_content content194)
     (no_gluten_bread bread21)
     (no_gluten_content content280)
     (allergic_gluten child238)
     (not_allergic_gluten child54)
     (waiting child238 table466)
     (waiting child54 table321)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child238)
     (served child54)
    )
  )
)
