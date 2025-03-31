; child-snack task with 2 children and 0.58 gluten factor 
; constant factor of 1.5
; random seed: 25254

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child313 child196 - child
    bread457 bread250 - bread-portion
    content413 content412 - content-portion
    tray324 tray465 tray366 - tray
    table229 table223 table201 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray324 kitchen)
     (at tray465 kitchen)
     (at tray366 kitchen)
     (at_kitchen_bread bread457)
     (at_kitchen_bread bread250)
     (at_kitchen_content content413)
     (at_kitchen_content content412)
     (no_gluten_bread bread457)
     (no_gluten_content content412)
     (allergic_gluten child196)
     (not_allergic_gluten child313)
     (waiting child313 table223)
     (waiting child196 table201)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child313)
     (served child196)
    )
  )
)
