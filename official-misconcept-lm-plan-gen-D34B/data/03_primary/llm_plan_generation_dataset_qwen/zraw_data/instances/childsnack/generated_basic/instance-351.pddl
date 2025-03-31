; child-snack task with 1 children and 0.2 gluten factor 
; constant factor of 1.9
; random seed: 952561

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child489 - child
    bread84 - bread-portion
    content144 - content-portion
    tray473 tray339 tray17 tray265 - tray
    table335 table101 table399 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray473 kitchen)
     (at tray339 kitchen)
     (at tray17 kitchen)
     (at tray265 kitchen)
     (at_kitchen_bread bread84)
     (at_kitchen_content content144)
     (not_allergic_gluten child489)
     (waiting child489 table399)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child489)
    )
  )
)
