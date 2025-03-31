; child-snack task with 1 children and 0.71 gluten factor 
; constant factor of 1.3
; random seed: 522310

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child110 - child
    bread176 - bread-portion
    content227 - content-portion
    tray151 tray152 tray116 tray155 - tray
    table20 table428 table420 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray151 kitchen)
     (at tray152 kitchen)
     (at tray116 kitchen)
     (at tray155 kitchen)
     (at_kitchen_bread bread176)
     (at_kitchen_content content227)
     (not_allergic_gluten child110)
     (waiting child110 table420)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child110)
    )
  )
)
