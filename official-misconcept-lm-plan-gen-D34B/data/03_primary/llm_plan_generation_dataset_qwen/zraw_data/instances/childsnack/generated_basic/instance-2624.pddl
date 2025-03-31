; child-snack task with 1 children and 0.48 gluten factor 
; constant factor of 1.7
; random seed: 677615

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child279 - child
    bread443 - bread-portion
    content426 - content-portion
    tray393 tray247 tray288 - tray
    table427 table44 table113 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray393 kitchen)
     (at tray247 kitchen)
     (at tray288 kitchen)
     (at_kitchen_bread bread443)
     (at_kitchen_content content426)
     (not_allergic_gluten child279)
     (waiting child279 table427)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child279)
    )
  )
)
