; child-snack task with 1 children and 0.36 gluten factor 
; constant factor of 1.3
; random seed: 322876

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child273 - child
    bread376 - bread-portion
    content447 - content-portion
    tray316 tray480 tray71 - tray
    table228 table384 table389 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray316 kitchen)
     (at tray480 kitchen)
     (at tray71 kitchen)
     (at_kitchen_bread bread376)
     (at_kitchen_content content447)
     (not_allergic_gluten child273)
     (waiting child273 table389)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child273)
    )
  )
)
