; child-snack task with 2 children and 0.8 gluten factor 
; constant factor of 1.7
; random seed: 580296

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child143 child179 - child
    bread360 bread392 - bread-portion
    content5 content27 - content-portion
    tray46 tray338 - tray
    table272 table296 table212 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray46 kitchen)
     (at tray338 kitchen)
     (at_kitchen_bread bread360)
     (at_kitchen_bread bread392)
     (at_kitchen_content content5)
     (at_kitchen_content content27)
     (no_gluten_bread bread360)
     (no_gluten_content content5)
     (allergic_gluten child143)
     (not_allergic_gluten child179)
     (waiting child143 table212)
     (waiting child179 table272)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child143)
     (served child179)
    )
  )
)
