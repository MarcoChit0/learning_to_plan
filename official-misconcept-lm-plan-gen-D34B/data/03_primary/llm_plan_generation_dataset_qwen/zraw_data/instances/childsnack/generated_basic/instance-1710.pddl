; child-snack task with 1 children and 0.36 gluten factor 
; constant factor of 1.3
; random seed: 372387

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child252 - child
    bread376 - bread-portion
    content83 - content-portion
    tray183 tray157 tray406 - tray
    table468 table38 table6 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray183 kitchen)
     (at tray157 kitchen)
     (at tray406 kitchen)
     (at_kitchen_bread bread376)
     (at_kitchen_content content83)
     (not_allergic_gluten child252)
     (waiting child252 table38)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child252)
    )
  )
)
