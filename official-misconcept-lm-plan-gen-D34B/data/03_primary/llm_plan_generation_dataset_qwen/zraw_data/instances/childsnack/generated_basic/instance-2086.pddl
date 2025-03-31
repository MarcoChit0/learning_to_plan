; child-snack task with 1 children and 0.62 gluten factor 
; constant factor of 1.5
; random seed: 738643

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child229 - child
    bread231 - bread-portion
    content16 - content-portion
    tray484 tray73 tray489 tray302 - tray
    table363 table117 table316 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray484 kitchen)
     (at tray73 kitchen)
     (at tray489 kitchen)
     (at tray302 kitchen)
     (at_kitchen_bread bread231)
     (at_kitchen_content content16)
     (not_allergic_gluten child229)
     (waiting child229 table316)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child229)
    )
  )
)
