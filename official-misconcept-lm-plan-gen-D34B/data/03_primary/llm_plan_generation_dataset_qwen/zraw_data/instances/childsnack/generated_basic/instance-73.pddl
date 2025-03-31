; child-snack task with 2 children and 0.36 gluten factor 
; constant factor of 1.9
; random seed: 213783

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child362 child315 - child
    bread262 bread129 - bread-portion
    content250 content315 - content-portion
    tray363 tray34 - tray
    table275 table272 table111 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray363 kitchen)
     (at tray34 kitchen)
     (at_kitchen_bread bread262)
     (at_kitchen_bread bread129)
     (at_kitchen_content content250)
     (at_kitchen_content content315)
     (not_allergic_gluten child315)
     (not_allergic_gluten child362)
     (waiting child362 table111)
     (waiting child315 table111)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child362)
     (served child315)
    )
  )
)
