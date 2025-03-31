; child-snack task with 1 children and 0.62 gluten factor 
; constant factor of 1.9
; random seed: 88287

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child140 - child
    bread201 - bread-portion
    content256 - content-portion
    tray3 tray306 tray56 - tray
    table393 table342 table288 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray3 kitchen)
     (at tray306 kitchen)
     (at tray56 kitchen)
     (at_kitchen_bread bread201)
     (at_kitchen_content content256)
     (not_allergic_gluten child140)
     (waiting child140 table288)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child140)
    )
  )
)
