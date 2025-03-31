; child-snack task with 1 children and 0.35 gluten factor 
; constant factor of 1.3
; random seed: 902743

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child359 - child
    bread311 - bread-portion
    content418 - content-portion
    tray440 tray494 tray333 tray236 - tray
    table286 table353 table369 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray440 kitchen)
     (at tray494 kitchen)
     (at tray333 kitchen)
     (at tray236 kitchen)
     (at_kitchen_bread bread311)
     (at_kitchen_content content418)
     (not_allergic_gluten child359)
     (waiting child359 table369)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child359)
    )
  )
)
