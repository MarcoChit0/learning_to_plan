; child-snack task with 1 children and 0.36 gluten factor 
; constant factor of 1.3
; random seed: 41271

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child153 - child
    bread3 - bread-portion
    content163 - content-portion
    tray110 tray48 tray426 - tray
    table354 table455 table454 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray110 kitchen)
     (at tray48 kitchen)
     (at tray426 kitchen)
     (at_kitchen_bread bread3)
     (at_kitchen_content content163)
     (not_allergic_gluten child153)
     (waiting child153 table354)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child153)
    )
  )
)
