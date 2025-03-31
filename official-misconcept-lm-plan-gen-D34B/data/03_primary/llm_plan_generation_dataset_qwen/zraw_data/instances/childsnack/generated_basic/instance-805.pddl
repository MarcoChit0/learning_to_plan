; child-snack task with 1 children and 0.49 gluten factor 
; constant factor of 1.7
; random seed: 983500

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child293 - child
    bread248 - bread-portion
    content34 - content-portion
    tray306 tray419 - tray
    table363 table116 table415 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray306 kitchen)
     (at tray419 kitchen)
     (at_kitchen_bread bread248)
     (at_kitchen_content content34)
     (not_allergic_gluten child293)
     (waiting child293 table116)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child293)
    )
  )
)
