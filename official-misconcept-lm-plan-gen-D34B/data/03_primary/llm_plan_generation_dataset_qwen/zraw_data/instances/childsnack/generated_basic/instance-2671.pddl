; child-snack task with 1 children and 0.35 gluten factor 
; constant factor of 1.3
; random seed: 138104

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child425 - child
    bread319 - bread-portion
    content193 - content-portion
    tray183 tray134 tray179 tray402 - tray
    table222 table74 table237 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray183 kitchen)
     (at tray134 kitchen)
     (at tray179 kitchen)
     (at tray402 kitchen)
     (at_kitchen_bread bread319)
     (at_kitchen_content content193)
     (not_allergic_gluten child425)
     (waiting child425 table237)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child425)
    )
  )
)
