; child-snack task with 1 children and 0.62 gluten factor 
; constant factor of 1.9
; random seed: 511427

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child147 - child
    bread33 - bread-portion
    content483 - content-portion
    tray307 tray266 tray282 - tray
    table402 table109 table367 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray307 kitchen)
     (at tray266 kitchen)
     (at tray282 kitchen)
     (at_kitchen_bread bread33)
     (at_kitchen_content content483)
     (not_allergic_gluten child147)
     (waiting child147 table402)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child147)
    )
  )
)
