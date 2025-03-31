; child-snack task with 1 children and 0.36 gluten factor 
; constant factor of 1.3
; random seed: 445136

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child394 - child
    bread123 - bread-portion
    content437 - content-portion
    tray43 tray205 tray409 - tray
    table132 table93 table131 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray43 kitchen)
     (at tray205 kitchen)
     (at tray409 kitchen)
     (at_kitchen_bread bread123)
     (at_kitchen_content content437)
     (not_allergic_gluten child394)
     (waiting child394 table131)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child394)
    )
  )
)
