; child-snack task with 1 children and 0.35 gluten factor 
; constant factor of 1.3
; random seed: 264353

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child431 - child
    bread73 - bread-portion
    content143 - content-portion
    tray486 tray87 tray172 tray393 - tray
    table411 table272 table150 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray486 kitchen)
     (at tray87 kitchen)
     (at tray172 kitchen)
     (at tray393 kitchen)
     (at_kitchen_bread bread73)
     (at_kitchen_content content143)
     (not_allergic_gluten child431)
     (waiting child431 table150)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child431)
    )
  )
)
