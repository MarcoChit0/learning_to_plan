; child-snack task with 2 children and 0.38 gluten factor 
; constant factor of 1.9
; random seed: 81115

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child0 child26 - child
    bread318 bread481 - bread-portion
    content257 content497 - content-portion
    tray271 - tray
    table12 table222 table422 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray271 kitchen)
     (at_kitchen_bread bread318)
     (at_kitchen_bread bread481)
     (at_kitchen_content content257)
     (at_kitchen_content content497)
     (not_allergic_gluten child0)
     (not_allergic_gluten child26)
     (waiting child0 table222)
     (waiting child26 table422)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child0)
     (served child26)
    )
  )
)
