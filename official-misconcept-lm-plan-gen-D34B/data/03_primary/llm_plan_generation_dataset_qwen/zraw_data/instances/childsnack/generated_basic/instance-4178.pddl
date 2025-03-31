; child-snack task with 2 children and 0.28 gluten factor 
; constant factor of 1.7
; random seed: 379374

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child306 child207 - child
    bread323 bread249 - bread-portion
    content135 content107 - content-portion
    tray20 - tray
    table447 table123 table313 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray20 kitchen)
     (at_kitchen_bread bread323)
     (at_kitchen_bread bread249)
     (at_kitchen_content content135)
     (at_kitchen_content content107)
     (not_allergic_gluten child207)
     (not_allergic_gluten child306)
     (waiting child306 table123)
     (waiting child207 table313)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child306)
     (served child207)
    )
  )
)
