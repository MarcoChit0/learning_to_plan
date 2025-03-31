; child-snack task with 1 children and 0.65 gluten factor 
; constant factor of 1.7
; random seed: 204162

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child263 - child
    bread80 - bread-portion
    content494 - content-portion
    tray394 tray450 tray439 - tray
    table14 table151 table160 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray394 kitchen)
     (at tray450 kitchen)
     (at tray439 kitchen)
     (at_kitchen_bread bread80)
     (at_kitchen_content content494)
     (not_allergic_gluten child263)
     (waiting child263 table14)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child263)
    )
  )
)
