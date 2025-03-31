; child-snack task with 1 children and 0.63 gluten factor 
; constant factor of 1.5
; random seed: 266380

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child246 - child
    bread316 - bread-portion
    content449 - content-portion
    tray425 - tray
    table119 table126 table368 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray425 kitchen)
     (at_kitchen_bread bread316)
     (at_kitchen_content content449)
     (not_allergic_gluten child246)
     (waiting child246 table368)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child246)
    )
  )
)
