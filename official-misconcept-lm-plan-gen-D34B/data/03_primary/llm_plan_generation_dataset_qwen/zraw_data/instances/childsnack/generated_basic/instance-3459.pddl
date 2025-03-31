; child-snack task with 1 children and 0.77 gluten factor 
; constant factor of 1.9
; random seed: 502938

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child381 - child
    bread288 - bread-portion
    content83 - content-portion
    tray157 tray335 tray284 tray475 - tray
    table496 table423 table255 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray157 kitchen)
     (at tray335 kitchen)
     (at tray284 kitchen)
     (at tray475 kitchen)
     (at_kitchen_bread bread288)
     (at_kitchen_content content83)
     (not_allergic_gluten child381)
     (waiting child381 table496)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child381)
    )
  )
)
