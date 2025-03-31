; child-snack task with 1 children and 0.47 gluten factor 
; constant factor of 1.7
; random seed: 666828

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child261 - child
    bread383 - bread-portion
    content231 - content-portion
    tray5 tray38 tray284 tray52 - tray
    table254 table384 table261 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray5 kitchen)
     (at tray38 kitchen)
     (at tray284 kitchen)
     (at tray52 kitchen)
     (at_kitchen_bread bread383)
     (at_kitchen_content content231)
     (not_allergic_gluten child261)
     (waiting child261 table384)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child261)
    )
  )
)
