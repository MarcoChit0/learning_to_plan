; child-snack task with 2 children and 0.55 gluten factor 
; constant factor of 1.7
; random seed: 802815

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child326 child77 - child
    bread270 bread132 - bread-portion
    content183 content408 - content-portion
    tray347 tray118 tray58 tray486 - tray
    table373 table451 table102 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray347 kitchen)
     (at tray118 kitchen)
     (at tray58 kitchen)
     (at tray486 kitchen)
     (at_kitchen_bread bread270)
     (at_kitchen_bread bread132)
     (at_kitchen_content content183)
     (at_kitchen_content content408)
     (no_gluten_bread bread270)
     (no_gluten_content content408)
     (allergic_gluten child326)
     (not_allergic_gluten child77)
     (waiting child326 table373)
     (waiting child77 table451)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child326)
     (served child77)
    )
  )
)
