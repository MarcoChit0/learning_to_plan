; child-snack task with 2 children and 0.78 gluten factor 
; constant factor of 1.9
; random seed: 668844

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child63 child411 - child
    bread449 bread43 - bread-portion
    content441 content327 - content-portion
    tray463 - tray
    table24 table300 table387 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray463 kitchen)
     (at_kitchen_bread bread449)
     (at_kitchen_bread bread43)
     (at_kitchen_content content441)
     (at_kitchen_content content327)
     (no_gluten_bread bread43)
     (no_gluten_content content327)
     (allergic_gluten child63)
     (not_allergic_gluten child411)
     (waiting child63 table387)
     (waiting child411 table387)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child63)
     (served child411)
    )
  )
)
