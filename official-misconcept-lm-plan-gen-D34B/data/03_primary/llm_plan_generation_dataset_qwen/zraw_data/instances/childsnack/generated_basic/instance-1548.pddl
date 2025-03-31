; child-snack task with 2 children and 0.49 gluten factor 
; constant factor of 1.7
; random seed: 448769

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child63 child109 - child
    bread107 bread265 - bread-portion
    content394 content311 - content-portion
    tray222 tray427 tray441 tray98 - tray
    table292 table275 table479 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray222 kitchen)
     (at tray427 kitchen)
     (at tray441 kitchen)
     (at tray98 kitchen)
     (at_kitchen_bread bread107)
     (at_kitchen_bread bread265)
     (at_kitchen_content content394)
     (at_kitchen_content content311)
     (not_allergic_gluten child63)
     (not_allergic_gluten child109)
     (waiting child63 table479)
     (waiting child109 table292)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child63)
     (served child109)
    )
  )
)
