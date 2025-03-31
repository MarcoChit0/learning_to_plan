; child-snack task with 2 children and 0.38 gluten factor 
; constant factor of 1.9
; random seed: 760981

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child181 child361 - child
    bread16 bread47 - bread-portion
    content91 content425 - content-portion
    tray32 - tray
    table33 table390 table208 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray32 kitchen)
     (at_kitchen_bread bread16)
     (at_kitchen_bread bread47)
     (at_kitchen_content content91)
     (at_kitchen_content content425)
     (not_allergic_gluten child181)
     (not_allergic_gluten child361)
     (waiting child181 table208)
     (waiting child361 table390)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child181)
     (served child361)
    )
  )
)
