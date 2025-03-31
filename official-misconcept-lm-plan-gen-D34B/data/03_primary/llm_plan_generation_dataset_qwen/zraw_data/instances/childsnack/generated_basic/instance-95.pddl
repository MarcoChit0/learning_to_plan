; child-snack task with 2 children and 0.36 gluten factor 
; constant factor of 1.9
; random seed: 974934

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child111 child304 - child
    bread374 bread119 - bread-portion
    content350 content359 - content-portion
    tray422 tray478 - tray
    table96 table89 table361 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray422 kitchen)
     (at tray478 kitchen)
     (at_kitchen_bread bread374)
     (at_kitchen_bread bread119)
     (at_kitchen_content content350)
     (at_kitchen_content content359)
     (not_allergic_gluten child111)
     (not_allergic_gluten child304)
     (waiting child111 table89)
     (waiting child304 table89)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child111)
     (served child304)
    )
  )
)
