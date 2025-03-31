; child-snack task with 2 children and 0.32 gluten factor 
; constant factor of 1.9
; random seed: 324612

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child293 child365 - child
    bread211 bread112 - bread-portion
    content487 content457 - content-portion
    tray377 tray324 - tray
    table480 table72 table67 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray377 kitchen)
     (at tray324 kitchen)
     (at_kitchen_bread bread211)
     (at_kitchen_bread bread112)
     (at_kitchen_content content487)
     (at_kitchen_content content457)
     (not_allergic_gluten child293)
     (not_allergic_gluten child365)
     (waiting child293 table67)
     (waiting child365 table480)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child293)
     (served child365)
    )
  )
)
