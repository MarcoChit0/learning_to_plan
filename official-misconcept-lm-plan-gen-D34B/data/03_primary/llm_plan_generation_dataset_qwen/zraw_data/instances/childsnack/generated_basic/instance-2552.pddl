; child-snack task with 1 children and 0.3 gluten factor 
; constant factor of 1.3
; random seed: 195081

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child32 - child
    bread345 - bread-portion
    content385 - content-portion
    tray199 tray136 tray18 - tray
    table0 table27 table305 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray199 kitchen)
     (at tray136 kitchen)
     (at tray18 kitchen)
     (at_kitchen_bread bread345)
     (at_kitchen_content content385)
     (not_allergic_gluten child32)
     (waiting child32 table27)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child32)
    )
  )
)
