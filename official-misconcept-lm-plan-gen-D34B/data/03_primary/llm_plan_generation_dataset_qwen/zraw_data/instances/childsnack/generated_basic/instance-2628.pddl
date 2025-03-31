; child-snack task with 1 children and 0.48 gluten factor 
; constant factor of 1.7
; random seed: 59138

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child499 - child
    bread92 - bread-portion
    content477 - content-portion
    tray54 tray151 tray479 - tray
    table368 table128 table350 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray54 kitchen)
     (at tray151 kitchen)
     (at tray479 kitchen)
     (at_kitchen_bread bread92)
     (at_kitchen_content content477)
     (not_allergic_gluten child499)
     (waiting child499 table350)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child499)
    )
  )
)
