; child-snack task with 3 children and 0.3 gluten factor 
; constant factor of 1.3
; random seed: 486526

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child113 child405 child65 - child
    bread496 bread32 bread28 - bread-portion
    content5 content496 content21 - content-portion
    tray499 - tray
    table354 table494 table473 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray499 kitchen)
     (at_kitchen_bread bread496)
     (at_kitchen_bread bread32)
     (at_kitchen_bread bread28)
     (at_kitchen_content content5)
     (at_kitchen_content content496)
     (at_kitchen_content content21)
     (not_allergic_gluten child65)
     (not_allergic_gluten child113)
     (not_allergic_gluten child405)
     (waiting child113 table494)
     (waiting child405 table473)
     (waiting child65 table494)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child113)
     (served child405)
     (served child65)
    )
  )
)
