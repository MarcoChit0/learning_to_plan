; child-snack task with 3 children and 0.23 gluten factor 
; constant factor of 1.9
; random seed: 630348

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child398 child493 child83 - child
    bread428 bread296 bread112 - bread-portion
    content272 content285 content82 - content-portion
    tray66 tray153 tray361 - tray
    table3 table30 table179 - place
    sandw1 sandw2 sandw3 sandw4 sandw5 sandw6 - sandwich
  )
  (:init
     (at tray66 kitchen)
     (at tray153 kitchen)
     (at tray361 kitchen)
     (at_kitchen_bread bread428)
     (at_kitchen_bread bread296)
     (at_kitchen_bread bread112)
     (at_kitchen_content content272)
     (at_kitchen_content content285)
     (at_kitchen_content content82)
     (not_allergic_gluten child398)
     (not_allergic_gluten child83)
     (not_allergic_gluten child493)
     (waiting child398 table3)
     (waiting child493 table3)
     (waiting child83 table3)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
     (notexist sandw5)
     (notexist sandw6)
  )
  (:goal
    (and
     (served child398)
     (served child493)
     (served child83)
    )
  )
)
