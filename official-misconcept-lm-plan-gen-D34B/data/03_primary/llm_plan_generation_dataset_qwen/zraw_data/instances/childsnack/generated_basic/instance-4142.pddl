; child-snack task with 2 children and 0.44 gluten factor 
; constant factor of 1.9
; random seed: 155741

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child234 child137 - child
    bread354 bread346 - bread-portion
    content267 content395 - content-portion
    tray5 tray364 tray390 - tray
    table25 table91 table272 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray5 kitchen)
     (at tray364 kitchen)
     (at tray390 kitchen)
     (at_kitchen_bread bread354)
     (at_kitchen_bread bread346)
     (at_kitchen_content content267)
     (at_kitchen_content content395)
     (not_allergic_gluten child137)
     (not_allergic_gluten child234)
     (waiting child234 table91)
     (waiting child137 table25)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child234)
     (served child137)
    )
  )
)
