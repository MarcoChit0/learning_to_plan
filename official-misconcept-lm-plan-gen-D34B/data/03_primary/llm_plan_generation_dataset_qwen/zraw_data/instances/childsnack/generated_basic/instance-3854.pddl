; child-snack task with 1 children and 0.47 gluten factor 
; constant factor of 1.7
; random seed: 692387

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child312 - child
    bread125 - bread-portion
    content189 - content-portion
    tray324 tray269 tray392 tray297 - tray
    table80 table68 table85 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray324 kitchen)
     (at tray269 kitchen)
     (at tray392 kitchen)
     (at tray297 kitchen)
     (at_kitchen_bread bread125)
     (at_kitchen_content content189)
     (not_allergic_gluten child312)
     (waiting child312 table68)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child312)
    )
  )
)
