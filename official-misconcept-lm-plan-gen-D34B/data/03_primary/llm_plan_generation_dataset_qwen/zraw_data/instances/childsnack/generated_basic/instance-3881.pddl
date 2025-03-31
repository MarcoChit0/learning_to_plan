; child-snack task with 1 children and 0.4 gluten factor 
; constant factor of 1.5
; random seed: 79193

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child306 - child
    bread427 - bread-portion
    content417 - content-portion
    tray39 tray147 - tray
    table284 table219 table286 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray39 kitchen)
     (at tray147 kitchen)
     (at_kitchen_bread bread427)
     (at_kitchen_content content417)
     (not_allergic_gluten child306)
     (waiting child306 table286)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child306)
    )
  )
)
