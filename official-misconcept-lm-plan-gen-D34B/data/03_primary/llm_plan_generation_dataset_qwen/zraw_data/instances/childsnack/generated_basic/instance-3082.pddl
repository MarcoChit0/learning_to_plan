; child-snack task with 2 children and 0.45 gluten factor 
; constant factor of 1.7
; random seed: 882943

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child27 child354 - child
    bread308 bread84 - bread-portion
    content68 content390 - content-portion
    tray271 tray25 tray334 - tray
    table409 table307 table247 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray271 kitchen)
     (at tray25 kitchen)
     (at tray334 kitchen)
     (at_kitchen_bread bread308)
     (at_kitchen_bread bread84)
     (at_kitchen_content content68)
     (at_kitchen_content content390)
     (not_allergic_gluten child27)
     (not_allergic_gluten child354)
     (waiting child27 table409)
     (waiting child354 table307)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child27)
     (served child354)
    )
  )
)
