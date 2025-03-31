; child-snack task with 2 children and 0.33 gluten factor 
; constant factor of 1.7
; random seed: 337158

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child147 child66 - child
    bread394 bread493 - bread-portion
    content485 content438 - content-portion
    tray47 tray386 tray426 tray372 - tray
    table358 table135 table348 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray47 kitchen)
     (at tray386 kitchen)
     (at tray426 kitchen)
     (at tray372 kitchen)
     (at_kitchen_bread bread394)
     (at_kitchen_bread bread493)
     (at_kitchen_content content485)
     (at_kitchen_content content438)
     (not_allergic_gluten child147)
     (not_allergic_gluten child66)
     (waiting child147 table348)
     (waiting child66 table348)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child147)
     (served child66)
    )
  )
)
