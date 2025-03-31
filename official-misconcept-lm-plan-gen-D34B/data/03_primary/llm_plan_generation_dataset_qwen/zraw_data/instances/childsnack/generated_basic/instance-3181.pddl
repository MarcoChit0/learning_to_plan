; child-snack task with 2 children and 0.48 gluten factor 
; constant factor of 1.9
; random seed: 905669

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child12 child259 - child
    bread489 bread32 - bread-portion
    content71 content73 - content-portion
    tray73 tray279 - tray
    table411 table209 table4 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray73 kitchen)
     (at tray279 kitchen)
     (at_kitchen_bread bread489)
     (at_kitchen_bread bread32)
     (at_kitchen_content content71)
     (at_kitchen_content content73)
     (not_allergic_gluten child12)
     (not_allergic_gluten child259)
     (waiting child12 table209)
     (waiting child259 table4)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child12)
     (served child259)
    )
  )
)
