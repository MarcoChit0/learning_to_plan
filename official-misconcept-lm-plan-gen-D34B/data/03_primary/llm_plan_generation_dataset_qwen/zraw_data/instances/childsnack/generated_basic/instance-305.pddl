; child-snack task with 1 children and 0.75 gluten factor 
; constant factor of 1.7
; random seed: 352216

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child237 - child
    bread453 - bread-portion
    content113 - content-portion
    tray170 - tray
    table423 table243 table444 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray170 kitchen)
     (at_kitchen_bread bread453)
     (at_kitchen_content content113)
     (not_allergic_gluten child237)
     (waiting child237 table423)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child237)
    )
  )
)
