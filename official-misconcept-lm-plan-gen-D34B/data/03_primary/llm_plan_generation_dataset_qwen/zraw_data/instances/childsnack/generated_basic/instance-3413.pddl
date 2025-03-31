; child-snack task with 1 children and 0.77 gluten factor 
; constant factor of 1.9
; random seed: 945865

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child394 - child
    bread67 - bread-portion
    content381 - content-portion
    tray170 tray392 tray144 tray452 - tray
    table213 table195 table123 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray170 kitchen)
     (at tray392 kitchen)
     (at tray144 kitchen)
     (at tray452 kitchen)
     (at_kitchen_bread bread67)
     (at_kitchen_content content381)
     (not_allergic_gluten child394)
     (waiting child394 table213)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child394)
    )
  )
)
