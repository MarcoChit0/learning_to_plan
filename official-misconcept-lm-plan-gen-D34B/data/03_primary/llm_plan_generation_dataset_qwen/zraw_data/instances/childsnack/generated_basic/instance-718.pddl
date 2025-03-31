; child-snack task with 1 children and 0.43 gluten factor 
; constant factor of 1.9
; random seed: 292381

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child2 - child
    bread417 - bread-portion
    content207 - content-portion
    tray418 tray37 tray28 tray442 - tray
    table408 table160 table151 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray418 kitchen)
     (at tray37 kitchen)
     (at tray28 kitchen)
     (at tray442 kitchen)
     (at_kitchen_bread bread417)
     (at_kitchen_content content207)
     (not_allergic_gluten child2)
     (waiting child2 table408)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child2)
    )
  )
)
