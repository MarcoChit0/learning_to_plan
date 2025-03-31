; child-snack task with 1 children and 0.77 gluten factor 
; constant factor of 1.9
; random seed: 915317

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child41 - child
    bread8 - bread-portion
    content420 - content-portion
    tray327 tray251 tray392 tray161 - tray
    table310 table11 table348 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray327 kitchen)
     (at tray251 kitchen)
     (at tray392 kitchen)
     (at tray161 kitchen)
     (at_kitchen_bread bread8)
     (at_kitchen_content content420)
     (not_allergic_gluten child41)
     (waiting child41 table11)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child41)
    )
  )
)
