; child-snack task with 3 children and 0.3 gluten factor 
; constant factor of 1.3
; random seed: 915018

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child181 child431 child346 - child
    bread72 bread426 bread469 - bread-portion
    content494 content370 content158 - content-portion
    tray50 - tray
    table96 table65 table469 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray50 kitchen)
     (at_kitchen_bread bread72)
     (at_kitchen_bread bread426)
     (at_kitchen_bread bread469)
     (at_kitchen_content content494)
     (at_kitchen_content content370)
     (at_kitchen_content content158)
     (not_allergic_gluten child431)
     (not_allergic_gluten child181)
     (not_allergic_gluten child346)
     (waiting child181 table65)
     (waiting child431 table96)
     (waiting child346 table469)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child181)
     (served child431)
     (served child346)
    )
  )
)
