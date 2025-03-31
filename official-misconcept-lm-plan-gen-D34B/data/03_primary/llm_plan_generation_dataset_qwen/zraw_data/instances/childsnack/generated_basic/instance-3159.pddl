; child-snack task with 2 children and 0.48 gluten factor 
; constant factor of 1.9
; random seed: 256677

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child451 child63 - child
    bread455 bread354 - bread-portion
    content466 content312 - content-portion
    tray125 tray297 - tray
    table168 table358 table308 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray125 kitchen)
     (at tray297 kitchen)
     (at_kitchen_bread bread455)
     (at_kitchen_bread bread354)
     (at_kitchen_content content466)
     (at_kitchen_content content312)
     (not_allergic_gluten child63)
     (not_allergic_gluten child451)
     (waiting child451 table168)
     (waiting child63 table308)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child451)
     (served child63)
    )
  )
)
