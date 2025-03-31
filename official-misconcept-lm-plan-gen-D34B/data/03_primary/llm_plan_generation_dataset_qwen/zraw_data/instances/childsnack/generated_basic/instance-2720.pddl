; child-snack task with 3 children and 0.23 gluten factor 
; constant factor of 1.9
; random seed: 117116

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child383 child479 child138 - child
    bread145 bread46 bread136 - bread-portion
    content203 content283 content219 - content-portion
    tray33 tray225 tray102 - tray
    table141 table402 table248 - place
    sandw1 sandw2 sandw3 sandw4 sandw5 sandw6 - sandwich
  )
  (:init
     (at tray33 kitchen)
     (at tray225 kitchen)
     (at tray102 kitchen)
     (at_kitchen_bread bread145)
     (at_kitchen_bread bread46)
     (at_kitchen_bread bread136)
     (at_kitchen_content content203)
     (at_kitchen_content content283)
     (at_kitchen_content content219)
     (not_allergic_gluten child479)
     (not_allergic_gluten child138)
     (not_allergic_gluten child383)
     (waiting child383 table141)
     (waiting child479 table402)
     (waiting child138 table248)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
     (notexist sandw5)
     (notexist sandw6)
  )
  (:goal
    (and
     (served child383)
     (served child479)
     (served child138)
    )
  )
)
