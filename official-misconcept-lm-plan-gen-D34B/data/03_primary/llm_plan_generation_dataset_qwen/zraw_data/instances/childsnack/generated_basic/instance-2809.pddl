; child-snack task with 2 children and 0.32 gluten factor 
; constant factor of 1.7
; random seed: 459599

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child221 child275 - child
    bread273 bread402 - bread-portion
    content294 content413 - content-portion
    tray35 tray0 - tray
    table180 table62 table429 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray35 kitchen)
     (at tray0 kitchen)
     (at_kitchen_bread bread273)
     (at_kitchen_bread bread402)
     (at_kitchen_content content294)
     (at_kitchen_content content413)
     (not_allergic_gluten child275)
     (not_allergic_gluten child221)
     (waiting child221 table429)
     (waiting child275 table429)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child221)
     (served child275)
    )
  )
)
