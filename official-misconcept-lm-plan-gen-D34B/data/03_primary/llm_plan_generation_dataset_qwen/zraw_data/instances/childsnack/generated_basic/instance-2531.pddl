; child-snack task with 1 children and 0.3 gluten factor 
; constant factor of 1.3
; random seed: 219595

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child439 - child
    bread40 - bread-portion
    content45 - content-portion
    tray23 tray388 tray365 - tray
    table478 table353 table328 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray23 kitchen)
     (at tray388 kitchen)
     (at tray365 kitchen)
     (at_kitchen_bread bread40)
     (at_kitchen_content content45)
     (not_allergic_gluten child439)
     (waiting child439 table353)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child439)
    )
  )
)
