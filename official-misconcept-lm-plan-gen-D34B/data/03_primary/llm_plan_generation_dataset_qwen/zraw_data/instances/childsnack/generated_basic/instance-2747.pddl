; child-snack task with 2 children and 0.77 gluten factor 
; constant factor of 1.3
; random seed: 430424

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child388 child61 - child
    bread292 bread208 - bread-portion
    content109 content291 - content-portion
    tray491 - tray
    table363 table16 table384 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray491 kitchen)
     (at_kitchen_bread bread292)
     (at_kitchen_bread bread208)
     (at_kitchen_content content109)
     (at_kitchen_content content291)
     (no_gluten_bread bread208)
     (no_gluten_content content109)
     (allergic_gluten child388)
     (not_allergic_gluten child61)
     (waiting child388 table363)
     (waiting child61 table363)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child388)
     (served child61)
    )
  )
)
