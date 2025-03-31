; child-snack task with 2 children and 0.33 gluten factor 
; constant factor of 1.7
; random seed: 768443

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child203 child224 - child
    bread18 bread304 - bread-portion
    content103 content371 - content-portion
    tray267 tray317 tray70 tray113 - tray
    table409 table237 table418 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray267 kitchen)
     (at tray317 kitchen)
     (at tray70 kitchen)
     (at tray113 kitchen)
     (at_kitchen_bread bread18)
     (at_kitchen_bread bread304)
     (at_kitchen_content content103)
     (at_kitchen_content content371)
     (not_allergic_gluten child203)
     (not_allergic_gluten child224)
     (waiting child203 table237)
     (waiting child224 table409)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child203)
     (served child224)
    )
  )
)
