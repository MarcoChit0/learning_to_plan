; child-snack task with 1 children and 0.36 gluten factor 
; constant factor of 1.3
; random seed: 473541

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child303 - child
    bread283 - bread-portion
    content341 - content-portion
    tray382 tray13 tray454 - tray
    table145 table3 table245 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray382 kitchen)
     (at tray13 kitchen)
     (at tray454 kitchen)
     (at_kitchen_bread bread283)
     (at_kitchen_content content341)
     (not_allergic_gluten child303)
     (waiting child303 table145)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child303)
    )
  )
)
