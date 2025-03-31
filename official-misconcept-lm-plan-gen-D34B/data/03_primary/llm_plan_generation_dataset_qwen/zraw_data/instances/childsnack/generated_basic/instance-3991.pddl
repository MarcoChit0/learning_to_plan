; child-snack task with 1 children and 0.62 gluten factor 
; constant factor of 1.9
; random seed: 611015

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child29 - child
    bread82 - bread-portion
    content193 - content-portion
    tray355 tray321 tray37 - tray
    table449 table171 table426 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray355 kitchen)
     (at tray321 kitchen)
     (at tray37 kitchen)
     (at_kitchen_bread bread82)
     (at_kitchen_content content193)
     (not_allergic_gluten child29)
     (waiting child29 table171)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child29)
    )
  )
)
