; child-snack task with 2 children and 0.32 gluten factor 
; constant factor of 1.9
; random seed: 787086

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child237 child183 - child
    bread431 bread141 - bread-portion
    content277 content335 - content-portion
    tray100 tray194 - tray
    table287 table71 table380 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray100 kitchen)
     (at tray194 kitchen)
     (at_kitchen_bread bread431)
     (at_kitchen_bread bread141)
     (at_kitchen_content content277)
     (at_kitchen_content content335)
     (not_allergic_gluten child237)
     (not_allergic_gluten child183)
     (waiting child237 table71)
     (waiting child183 table287)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child237)
     (served child183)
    )
  )
)
