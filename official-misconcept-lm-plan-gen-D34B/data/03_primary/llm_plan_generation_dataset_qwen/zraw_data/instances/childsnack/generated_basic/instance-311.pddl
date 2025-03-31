; child-snack task with 1 children and 0.75 gluten factor 
; constant factor of 1.7
; random seed: 425605

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child296 - child
    bread270 - bread-portion
    content156 - content-portion
    tray295 - tray
    table406 table360 table130 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray295 kitchen)
     (at_kitchen_bread bread270)
     (at_kitchen_content content156)
     (not_allergic_gluten child296)
     (waiting child296 table360)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child296)
    )
  )
)
