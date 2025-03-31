; child-snack task with 2 children and 0.34 gluten factor 
; constant factor of 1.5
; random seed: 592867

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child456 child208 - child
    bread73 bread100 - bread-portion
    content323 content169 - content-portion
    tray186 tray170 tray288 - tray
    table203 table327 table112 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray186 kitchen)
     (at tray170 kitchen)
     (at tray288 kitchen)
     (at_kitchen_bread bread73)
     (at_kitchen_bread bread100)
     (at_kitchen_content content323)
     (at_kitchen_content content169)
     (not_allergic_gluten child456)
     (not_allergic_gluten child208)
     (waiting child456 table203)
     (waiting child208 table203)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child456)
     (served child208)
    )
  )
)
