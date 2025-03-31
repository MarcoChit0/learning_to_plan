; child-snack task with 2 children and 0.36 gluten factor 
; constant factor of 1.9
; random seed: 923027

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child34 child468 - child
    bread297 bread73 - bread-portion
    content363 content1 - content-portion
    tray373 tray429 - tray
    table32 table248 table427 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray373 kitchen)
     (at tray429 kitchen)
     (at_kitchen_bread bread297)
     (at_kitchen_bread bread73)
     (at_kitchen_content content363)
     (at_kitchen_content content1)
     (not_allergic_gluten child34)
     (not_allergic_gluten child468)
     (waiting child34 table248)
     (waiting child468 table427)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child34)
     (served child468)
    )
  )
)
