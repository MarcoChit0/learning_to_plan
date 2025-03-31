; child-snack task with 2 children and 0.38 gluten factor 
; constant factor of 1.9
; random seed: 930982

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child39 child306 - child
    bread214 bread483 - bread-portion
    content69 content390 - content-portion
    tray128 - tray
    table499 table277 table497 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray128 kitchen)
     (at_kitchen_bread bread214)
     (at_kitchen_bread bread483)
     (at_kitchen_content content69)
     (at_kitchen_content content390)
     (not_allergic_gluten child39)
     (not_allergic_gluten child306)
     (waiting child39 table499)
     (waiting child306 table497)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child39)
     (served child306)
    )
  )
)
