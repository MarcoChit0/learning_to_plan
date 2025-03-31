; child-snack task with 2 children and 0.33 gluten factor 
; constant factor of 1.7
; random seed: 176488

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child24 child70 - child
    bread92 bread120 - bread-portion
    content170 content211 - content-portion
    tray6 tray419 tray497 tray490 - tray
    table237 table275 table201 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray6 kitchen)
     (at tray419 kitchen)
     (at tray497 kitchen)
     (at tray490 kitchen)
     (at_kitchen_bread bread92)
     (at_kitchen_bread bread120)
     (at_kitchen_content content170)
     (at_kitchen_content content211)
     (not_allergic_gluten child70)
     (not_allergic_gluten child24)
     (waiting child24 table275)
     (waiting child70 table237)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child24)
     (served child70)
    )
  )
)
