; child-snack task with 2 children and 0.63 gluten factor 
; constant factor of 1.7
; random seed: 8498

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child39 child466 - child
    bread204 bread419 - bread-portion
    content476 content462 - content-portion
    tray435 tray345 - tray
    table124 table432 table291 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray435 kitchen)
     (at tray345 kitchen)
     (at_kitchen_bread bread204)
     (at_kitchen_bread bread419)
     (at_kitchen_content content476)
     (at_kitchen_content content462)
     (no_gluten_bread bread204)
     (no_gluten_content content476)
     (allergic_gluten child466)
     (not_allergic_gluten child39)
     (waiting child39 table432)
     (waiting child466 table291)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child39)
     (served child466)
    )
  )
)
