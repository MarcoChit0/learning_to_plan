; child-snack task with 2 children and 0.38 gluten factor 
; constant factor of 1.9
; random seed: 567475

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child179 child227 - child
    bread342 bread481 - bread-portion
    content369 content163 - content-portion
    tray285 - tray
    table344 table418 table240 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray285 kitchen)
     (at_kitchen_bread bread342)
     (at_kitchen_bread bread481)
     (at_kitchen_content content369)
     (at_kitchen_content content163)
     (not_allergic_gluten child179)
     (not_allergic_gluten child227)
     (waiting child179 table240)
     (waiting child227 table418)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child179)
     (served child227)
    )
  )
)
