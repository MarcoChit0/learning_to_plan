; child-snack task with 1 children and 0.2 gluten factor 
; constant factor of 1.9
; random seed: 992463

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child458 - child
    bread121 - bread-portion
    content194 - content-portion
    tray283 tray460 tray296 tray83 - tray
    table405 table478 table17 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray283 kitchen)
     (at tray460 kitchen)
     (at tray296 kitchen)
     (at tray83 kitchen)
     (at_kitchen_bread bread121)
     (at_kitchen_content content194)
     (not_allergic_gluten child458)
     (waiting child458 table17)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child458)
    )
  )
)
