; child-snack task with 1 children and 0.47 gluten factor 
; constant factor of 1.7
; random seed: 240444

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child477 - child
    bread274 - bread-portion
    content487 - content-portion
    tray349 tray73 tray237 tray337 - tray
    table63 table383 table412 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray349 kitchen)
     (at tray73 kitchen)
     (at tray237 kitchen)
     (at tray337 kitchen)
     (at_kitchen_bread bread274)
     (at_kitchen_content content487)
     (not_allergic_gluten child477)
     (waiting child477 table412)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child477)
    )
  )
)
