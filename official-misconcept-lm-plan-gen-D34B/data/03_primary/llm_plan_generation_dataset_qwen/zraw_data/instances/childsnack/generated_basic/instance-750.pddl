; child-snack task with 2 children and 0.64 gluten factor 
; constant factor of 1.9
; random seed: 678214

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child108 child346 - child
    bread239 bread417 - bread-portion
    content201 content363 - content-portion
    tray239 tray170 - tray
    table498 table473 table249 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray239 kitchen)
     (at tray170 kitchen)
     (at_kitchen_bread bread239)
     (at_kitchen_bread bread417)
     (at_kitchen_content content201)
     (at_kitchen_content content363)
     (no_gluten_bread bread417)
     (no_gluten_content content201)
     (allergic_gluten child346)
     (not_allergic_gluten child108)
     (waiting child108 table473)
     (waiting child346 table249)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child108)
     (served child346)
    )
  )
)
