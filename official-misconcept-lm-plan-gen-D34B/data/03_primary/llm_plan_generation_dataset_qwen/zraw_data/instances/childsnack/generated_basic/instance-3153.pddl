; child-snack task with 2 children and 0.48 gluten factor 
; constant factor of 1.9
; random seed: 532551

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child61 child375 - child
    bread400 bread375 - bread-portion
    content242 content192 - content-portion
    tray387 tray492 - tray
    table352 table473 table80 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray387 kitchen)
     (at tray492 kitchen)
     (at_kitchen_bread bread400)
     (at_kitchen_bread bread375)
     (at_kitchen_content content242)
     (at_kitchen_content content192)
     (not_allergic_gluten child61)
     (not_allergic_gluten child375)
     (waiting child61 table473)
     (waiting child375 table80)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child61)
     (served child375)
    )
  )
)
