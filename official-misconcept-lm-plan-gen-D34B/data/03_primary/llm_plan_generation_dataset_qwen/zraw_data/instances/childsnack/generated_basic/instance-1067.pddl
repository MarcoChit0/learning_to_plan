; child-snack task with 2 children and 0.41 gluten factor 
; constant factor of 1.3
; random seed: 362399

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child406 child300 - child
    bread258 bread66 - bread-portion
    content317 content7 - content-portion
    tray433 tray180 tray392 - tray
    table383 table6 table354 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray433 kitchen)
     (at tray180 kitchen)
     (at tray392 kitchen)
     (at_kitchen_bread bread258)
     (at_kitchen_bread bread66)
     (at_kitchen_content content317)
     (at_kitchen_content content7)
     (not_allergic_gluten child406)
     (not_allergic_gluten child300)
     (waiting child406 table383)
     (waiting child300 table354)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child406)
     (served child300)
    )
  )
)
