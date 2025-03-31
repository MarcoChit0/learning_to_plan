; child-snack task with 2 children and 0.44 gluten factor 
; constant factor of 1.9
; random seed: 3462

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child104 child31 - child
    bread303 bread109 - bread-portion
    content427 content432 - content-portion
    tray42 tray132 tray20 - tray
    table181 table350 table250 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray42 kitchen)
     (at tray132 kitchen)
     (at tray20 kitchen)
     (at_kitchen_bread bread303)
     (at_kitchen_bread bread109)
     (at_kitchen_content content427)
     (at_kitchen_content content432)
     (not_allergic_gluten child104)
     (not_allergic_gluten child31)
     (waiting child104 table250)
     (waiting child31 table181)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child104)
     (served child31)
    )
  )
)
