; child-snack task with 1 children and 0.65 gluten factor 
; constant factor of 1.7
; random seed: 971286

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child325 - child
    bread136 - bread-portion
    content67 - content-portion
    tray93 tray345 tray118 - tray
    table28 table440 table104 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray93 kitchen)
     (at tray345 kitchen)
     (at tray118 kitchen)
     (at_kitchen_bread bread136)
     (at_kitchen_content content67)
     (not_allergic_gluten child325)
     (waiting child325 table28)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child325)
    )
  )
)
