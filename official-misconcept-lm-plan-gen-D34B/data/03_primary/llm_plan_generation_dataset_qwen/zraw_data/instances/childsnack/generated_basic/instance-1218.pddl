; child-snack task with 3 children and 0.33 gluten factor 
; constant factor of 1.7
; random seed: 491318

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child496 child106 child319 - child
    bread348 bread431 bread384 - bread-portion
    content352 content48 content295 - content-portion
    tray217 tray121 tray272 - tray
    table477 table71 table272 - place
    sandw1 sandw2 sandw3 sandw4 sandw5 sandw6 - sandwich
  )
  (:init
     (at tray217 kitchen)
     (at tray121 kitchen)
     (at tray272 kitchen)
     (at_kitchen_bread bread348)
     (at_kitchen_bread bread431)
     (at_kitchen_bread bread384)
     (at_kitchen_content content352)
     (at_kitchen_content content48)
     (at_kitchen_content content295)
     (not_allergic_gluten child106)
     (not_allergic_gluten child319)
     (not_allergic_gluten child496)
     (waiting child496 table477)
     (waiting child106 table272)
     (waiting child319 table477)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
     (notexist sandw5)
     (notexist sandw6)
  )
  (:goal
    (and
     (served child496)
     (served child106)
     (served child319)
    )
  )
)
