; child-snack task with 1 children and 0.65 gluten factor 
; constant factor of 1.7
; random seed: 517016

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child116 - child
    bread117 - bread-portion
    content339 - content-portion
    tray300 tray203 tray171 - tray
    table21 table409 table105 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray300 kitchen)
     (at tray203 kitchen)
     (at tray171 kitchen)
     (at_kitchen_bread bread117)
     (at_kitchen_content content339)
     (not_allergic_gluten child116)
     (waiting child116 table21)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child116)
    )
  )
)
