; child-snack task with 1 children and 0.79 gluten factor 
; constant factor of 1.9
; random seed: 261645

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child100 - child
    bread75 - bread-portion
    content479 - content-portion
    tray285 tray0 tray290 - tray
    table203 table6 table156 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray285 kitchen)
     (at tray0 kitchen)
     (at tray290 kitchen)
     (at_kitchen_bread bread75)
     (at_kitchen_content content479)
     (not_allergic_gluten child100)
     (waiting child100 table156)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child100)
    )
  )
)
