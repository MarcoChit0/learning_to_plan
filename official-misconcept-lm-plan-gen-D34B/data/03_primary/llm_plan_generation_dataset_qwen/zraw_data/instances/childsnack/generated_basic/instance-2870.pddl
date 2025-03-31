; child-snack task with 1 children and 0.23 gluten factor 
; constant factor of 1.7
; random seed: 998581

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child271 - child
    bread330 - bread-portion
    content134 - content-portion
    tray469 tray234 - tray
    table71 table296 table29 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray469 kitchen)
     (at tray234 kitchen)
     (at_kitchen_bread bread330)
     (at_kitchen_content content134)
     (not_allergic_gluten child271)
     (waiting child271 table29)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child271)
    )
  )
)
