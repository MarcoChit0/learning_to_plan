; child-snack task with 2 children and 0.54 gluten factor 
; constant factor of 1.7
; random seed: 213953

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child179 child490 - child
    bread43 bread295 - bread-portion
    content311 content217 - content-portion
    tray73 - tray
    table425 table181 table140 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray73 kitchen)
     (at_kitchen_bread bread43)
     (at_kitchen_bread bread295)
     (at_kitchen_content content311)
     (at_kitchen_content content217)
     (no_gluten_bread bread295)
     (no_gluten_content content311)
     (allergic_gluten child179)
     (not_allergic_gluten child490)
     (waiting child179 table181)
     (waiting child490 table425)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child179)
     (served child490)
    )
  )
)
