; child-snack task with 1 children and 0.49 gluten factor 
; constant factor of 1.7
; random seed: 920186

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child89 - child
    bread71 - bread-portion
    content126 - content-portion
    tray481 tray203 - tray
    table353 table329 table104 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray481 kitchen)
     (at tray203 kitchen)
     (at_kitchen_bread bread71)
     (at_kitchen_content content126)
     (not_allergic_gluten child89)
     (waiting child89 table353)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child89)
    )
  )
)
