; child-snack task with 2 children and 0.79 gluten factor 
; constant factor of 1.9
; random seed: 586878

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child1 child61 - child
    bread472 bread158 - bread-portion
    content297 content4 - content-portion
    tray230 tray482 tray43 tray265 - tray
    table27 table406 table147 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray230 kitchen)
     (at tray482 kitchen)
     (at tray43 kitchen)
     (at tray265 kitchen)
     (at_kitchen_bread bread472)
     (at_kitchen_bread bread158)
     (at_kitchen_content content297)
     (at_kitchen_content content4)
     (no_gluten_bread bread472)
     (no_gluten_content content297)
     (allergic_gluten child1)
     (not_allergic_gluten child61)
     (waiting child1 table406)
     (waiting child61 table406)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child1)
     (served child61)
    )
  )
)
