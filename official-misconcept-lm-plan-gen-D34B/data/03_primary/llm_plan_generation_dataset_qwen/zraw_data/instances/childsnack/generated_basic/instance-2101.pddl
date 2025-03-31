; child-snack task with 1 children and 0.62 gluten factor 
; constant factor of 1.5
; random seed: 891823

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child217 - child
    bread438 - bread-portion
    content116 - content-portion
    tray470 tray208 tray498 tray317 - tray
    table414 table368 table345 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray470 kitchen)
     (at tray208 kitchen)
     (at tray498 kitchen)
     (at tray317 kitchen)
     (at_kitchen_bread bread438)
     (at_kitchen_content content116)
     (not_allergic_gluten child217)
     (waiting child217 table368)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child217)
    )
  )
)
