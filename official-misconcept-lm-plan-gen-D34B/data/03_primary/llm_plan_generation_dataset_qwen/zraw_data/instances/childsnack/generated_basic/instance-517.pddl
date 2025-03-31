; child-snack task with 2 children and 0.32 gluten factor 
; constant factor of 1.9
; random seed: 155526

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child490 child349 - child
    bread48 bread139 - bread-portion
    content311 content257 - content-portion
    tray62 tray154 - tray
    table384 table464 table34 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray62 kitchen)
     (at tray154 kitchen)
     (at_kitchen_bread bread48)
     (at_kitchen_bread bread139)
     (at_kitchen_content content311)
     (at_kitchen_content content257)
     (not_allergic_gluten child490)
     (not_allergic_gluten child349)
     (waiting child490 table34)
     (waiting child349 table384)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child490)
     (served child349)
    )
  )
)
