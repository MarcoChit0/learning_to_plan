; child-snack task with 1 children and 0.36 gluten factor 
; constant factor of 1.3
; random seed: 562517

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child230 - child
    bread463 - bread-portion
    content141 - content-portion
    tray325 tray426 tray38 - tray
    table388 table427 table26 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray325 kitchen)
     (at tray426 kitchen)
     (at tray38 kitchen)
     (at_kitchen_bread bread463)
     (at_kitchen_content content141)
     (not_allergic_gluten child230)
     (waiting child230 table427)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child230)
    )
  )
)
