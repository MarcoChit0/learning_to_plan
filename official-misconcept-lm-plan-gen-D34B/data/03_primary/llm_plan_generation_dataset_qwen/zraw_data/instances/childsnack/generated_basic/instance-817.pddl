; child-snack task with 1 children and 0.49 gluten factor 
; constant factor of 1.7
; random seed: 338129

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child334 - child
    bread289 - bread-portion
    content365 - content-portion
    tray399 tray427 - tray
    table126 table402 table0 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray399 kitchen)
     (at tray427 kitchen)
     (at_kitchen_bread bread289)
     (at_kitchen_content content365)
     (not_allergic_gluten child334)
     (waiting child334 table402)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child334)
    )
  )
)
