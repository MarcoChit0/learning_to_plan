; child-snack task with 1 children and 0.77 gluten factor 
; constant factor of 1.9
; random seed: 617900

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child378 - child
    bread237 - bread-portion
    content6 - content-portion
    tray445 tray20 tray467 tray28 - tray
    table240 table38 table336 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray445 kitchen)
     (at tray20 kitchen)
     (at tray467 kitchen)
     (at tray28 kitchen)
     (at_kitchen_bread bread237)
     (at_kitchen_content content6)
     (not_allergic_gluten child378)
     (waiting child378 table240)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child378)
    )
  )
)
