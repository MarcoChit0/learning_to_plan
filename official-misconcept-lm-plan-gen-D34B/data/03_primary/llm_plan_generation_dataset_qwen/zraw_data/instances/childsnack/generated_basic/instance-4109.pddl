; child-snack task with 1 children and 0.79 gluten factor 
; constant factor of 1.9
; random seed: 772540

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child318 - child
    bread16 - bread-portion
    content286 - content-portion
    tray45 tray425 tray414 - tray
    table88 table427 table202 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray45 kitchen)
     (at tray425 kitchen)
     (at tray414 kitchen)
     (at_kitchen_bread bread16)
     (at_kitchen_content content286)
     (not_allergic_gluten child318)
     (waiting child318 table88)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child318)
    )
  )
)
