; child-snack task with 2 children and 0.44 gluten factor 
; constant factor of 1.9
; random seed: 508807

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child129 child258 - child
    bread42 bread425 - bread-portion
    content274 content417 - content-portion
    tray112 tray6 tray302 - tray
    table195 table269 table81 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray112 kitchen)
     (at tray6 kitchen)
     (at tray302 kitchen)
     (at_kitchen_bread bread42)
     (at_kitchen_bread bread425)
     (at_kitchen_content content274)
     (at_kitchen_content content417)
     (not_allergic_gluten child258)
     (not_allergic_gluten child129)
     (waiting child129 table195)
     (waiting child258 table81)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child129)
     (served child258)
    )
  )
)
