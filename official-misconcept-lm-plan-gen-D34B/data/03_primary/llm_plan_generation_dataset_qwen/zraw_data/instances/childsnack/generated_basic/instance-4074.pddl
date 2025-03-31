; child-snack task with 1 children and 0.79 gluten factor 
; constant factor of 1.9
; random seed: 105982

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child271 - child
    bread12 - bread-portion
    content454 - content-portion
    tray268 tray101 tray200 - tray
    table317 table373 table429 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray268 kitchen)
     (at tray101 kitchen)
     (at tray200 kitchen)
     (at_kitchen_bread bread12)
     (at_kitchen_content content454)
     (not_allergic_gluten child271)
     (waiting child271 table373)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child271)
    )
  )
)
