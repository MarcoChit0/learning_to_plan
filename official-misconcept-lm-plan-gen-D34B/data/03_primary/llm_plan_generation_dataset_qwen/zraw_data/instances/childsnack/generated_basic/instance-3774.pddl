; child-snack task with 1 children and 0.46 gluten factor 
; constant factor of 1.9
; random seed: 479997

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child240 - child
    bread120 - bread-portion
    content299 - content-portion
    tray497 - tray
    table334 table210 table58 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray497 kitchen)
     (at_kitchen_bread bread120)
     (at_kitchen_content content299)
     (not_allergic_gluten child240)
     (waiting child240 table334)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child240)
    )
  )
)
