; child-snack task with 1 children and 0.35 gluten factor 
; constant factor of 1.3
; random seed: 619766

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child310 - child
    bread377 - bread-portion
    content304 - content-portion
    tray434 tray427 tray162 tray6 - tray
    table446 table348 table404 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray434 kitchen)
     (at tray427 kitchen)
     (at tray162 kitchen)
     (at tray6 kitchen)
     (at_kitchen_bread bread377)
     (at_kitchen_content content304)
     (not_allergic_gluten child310)
     (waiting child310 table446)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child310)
    )
  )
)
