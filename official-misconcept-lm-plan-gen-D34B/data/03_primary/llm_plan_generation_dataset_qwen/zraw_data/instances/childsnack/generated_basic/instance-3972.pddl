; child-snack task with 1 children and 0.62 gluten factor 
; constant factor of 1.9
; random seed: 338238

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child83 - child
    bread39 - bread-portion
    content75 - content-portion
    tray83 tray272 tray264 - tray
    table459 table35 table481 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray83 kitchen)
     (at tray272 kitchen)
     (at tray264 kitchen)
     (at_kitchen_bread bread39)
     (at_kitchen_content content75)
     (not_allergic_gluten child83)
     (waiting child83 table459)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child83)
    )
  )
)
