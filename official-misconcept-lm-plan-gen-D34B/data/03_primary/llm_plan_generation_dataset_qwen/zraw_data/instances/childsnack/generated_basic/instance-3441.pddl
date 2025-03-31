; child-snack task with 1 children and 0.77 gluten factor 
; constant factor of 1.9
; random seed: 193552

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child137 - child
    bread127 - bread-portion
    content309 - content-portion
    tray449 tray115 tray83 tray434 - tray
    table372 table367 table457 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray449 kitchen)
     (at tray115 kitchen)
     (at tray83 kitchen)
     (at tray434 kitchen)
     (at_kitchen_bread bread127)
     (at_kitchen_content content309)
     (not_allergic_gluten child137)
     (waiting child137 table372)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child137)
    )
  )
)
