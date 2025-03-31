; child-snack task with 2 children and 0.41 gluten factor 
; constant factor of 1.9
; random seed: 274687

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child160 child205 - child
    bread65 bread331 - bread-portion
    content466 content330 - content-portion
    tray420 tray168 tray28 tray354 - tray
    table251 table494 table213 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray420 kitchen)
     (at tray168 kitchen)
     (at tray28 kitchen)
     (at tray354 kitchen)
     (at_kitchen_bread bread65)
     (at_kitchen_bread bread331)
     (at_kitchen_content content466)
     (at_kitchen_content content330)
     (not_allergic_gluten child205)
     (not_allergic_gluten child160)
     (waiting child160 table251)
     (waiting child205 table494)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child160)
     (served child205)
    )
  )
)
