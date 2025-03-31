; child-snack task with 2 children and 0.49 gluten factor 
; constant factor of 1.7
; random seed: 763018

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child216 child376 - child
    bread219 bread427 - bread-portion
    content297 content450 - content-portion
    tray343 tray189 tray299 tray487 - tray
    table333 table479 table388 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray343 kitchen)
     (at tray189 kitchen)
     (at tray299 kitchen)
     (at tray487 kitchen)
     (at_kitchen_bread bread219)
     (at_kitchen_bread bread427)
     (at_kitchen_content content297)
     (at_kitchen_content content450)
     (not_allergic_gluten child376)
     (not_allergic_gluten child216)
     (waiting child216 table388)
     (waiting child376 table333)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child216)
     (served child376)
    )
  )
)
