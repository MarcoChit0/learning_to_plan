; child-snack task with 2 children and 0.36 gluten factor 
; constant factor of 1.9
; random seed: 821740

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child325 child89 - child
    bread243 bread223 - bread-portion
    content465 content80 - content-portion
    tray18 tray375 - tray
    table136 table106 table30 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray18 kitchen)
     (at tray375 kitchen)
     (at_kitchen_bread bread243)
     (at_kitchen_bread bread223)
     (at_kitchen_content content465)
     (at_kitchen_content content80)
     (not_allergic_gluten child325)
     (not_allergic_gluten child89)
     (waiting child325 table106)
     (waiting child89 table30)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child325)
     (served child89)
    )
  )
)
