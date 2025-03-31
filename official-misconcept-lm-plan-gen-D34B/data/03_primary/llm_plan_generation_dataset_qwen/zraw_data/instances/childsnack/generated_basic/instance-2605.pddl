; child-snack task with 1 children and 0.48 gluten factor 
; constant factor of 1.7
; random seed: 989495

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child131 - child
    bread169 - bread-portion
    content354 - content-portion
    tray128 tray269 tray257 - tray
    table301 table148 table323 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray128 kitchen)
     (at tray269 kitchen)
     (at tray257 kitchen)
     (at_kitchen_bread bread169)
     (at_kitchen_content content354)
     (not_allergic_gluten child131)
     (waiting child131 table148)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child131)
    )
  )
)
