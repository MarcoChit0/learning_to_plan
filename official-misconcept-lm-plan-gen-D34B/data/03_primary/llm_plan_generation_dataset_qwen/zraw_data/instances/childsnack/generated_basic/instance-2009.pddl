; child-snack task with 2 children and 0.34 gluten factor 
; constant factor of 1.5
; random seed: 801214

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child8 child294 - child
    bread193 bread141 - bread-portion
    content405 content359 - content-portion
    tray197 tray429 tray198 - tray
    table77 table128 table491 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray197 kitchen)
     (at tray429 kitchen)
     (at tray198 kitchen)
     (at_kitchen_bread bread193)
     (at_kitchen_bread bread141)
     (at_kitchen_content content405)
     (at_kitchen_content content359)
     (not_allergic_gluten child294)
     (not_allergic_gluten child8)
     (waiting child8 table491)
     (waiting child294 table77)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child8)
     (served child294)
    )
  )
)
