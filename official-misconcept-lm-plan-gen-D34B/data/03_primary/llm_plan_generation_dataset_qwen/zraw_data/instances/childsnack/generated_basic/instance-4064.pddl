; child-snack task with 1 children and 0.79 gluten factor 
; constant factor of 1.9
; random seed: 134197

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child396 - child
    bread397 - bread-portion
    content397 - content-portion
    tray151 tray335 tray445 - tray
    table225 table392 table254 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray151 kitchen)
     (at tray335 kitchen)
     (at tray445 kitchen)
     (at_kitchen_bread bread397)
     (at_kitchen_content content397)
     (not_allergic_gluten child396)
     (waiting child396 table392)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child396)
    )
  )
)
