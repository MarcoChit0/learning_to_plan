; child-snack task with 2 children and 0.41 gluten factor 
; constant factor of 1.9
; random seed: 227557

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child355 child305 - child
    bread349 bread236 - bread-portion
    content281 content104 - content-portion
    tray51 tray200 tray184 tray161 - tray
    table423 table239 table109 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray51 kitchen)
     (at tray200 kitchen)
     (at tray184 kitchen)
     (at tray161 kitchen)
     (at_kitchen_bread bread349)
     (at_kitchen_bread bread236)
     (at_kitchen_content content281)
     (at_kitchen_content content104)
     (not_allergic_gluten child355)
     (not_allergic_gluten child305)
     (waiting child355 table239)
     (waiting child305 table239)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child355)
     (served child305)
    )
  )
)
