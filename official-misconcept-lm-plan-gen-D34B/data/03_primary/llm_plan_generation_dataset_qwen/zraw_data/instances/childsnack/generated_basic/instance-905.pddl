; child-snack task with 3 children and 0.31 gluten factor 
; constant factor of 1.5
; random seed: 10022

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child373 child91 child430 - child
    bread243 bread463 bread52 - bread-portion
    content409 content476 content495 - content-portion
    tray392 - tray
    table87 table496 table402 - place
    sandw1 sandw2 sandw3 sandw4 sandw5 - sandwich
  )
  (:init
     (at tray392 kitchen)
     (at_kitchen_bread bread243)
     (at_kitchen_bread bread463)
     (at_kitchen_bread bread52)
     (at_kitchen_content content409)
     (at_kitchen_content content476)
     (at_kitchen_content content495)
     (not_allergic_gluten child430)
     (not_allergic_gluten child91)
     (not_allergic_gluten child373)
     (waiting child373 table402)
     (waiting child91 table496)
     (waiting child430 table402)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
     (notexist sandw5)
  )
  (:goal
    (and
     (served child373)
     (served child91)
     (served child430)
    )
  )
)
