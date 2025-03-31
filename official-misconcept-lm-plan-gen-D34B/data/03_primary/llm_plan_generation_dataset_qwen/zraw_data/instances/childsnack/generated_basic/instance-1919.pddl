; child-snack task with 1 children and 0.65 gluten factor 
; constant factor of 1.7
; random seed: 621140

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child315 - child
    bread42 - bread-portion
    content113 - content-portion
    tray280 tray266 tray68 - tray
    table409 table67 table150 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray280 kitchen)
     (at tray266 kitchen)
     (at tray68 kitchen)
     (at_kitchen_bread bread42)
     (at_kitchen_content content113)
     (not_allergic_gluten child315)
     (waiting child315 table150)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child315)
    )
  )
)
