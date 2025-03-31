; child-snack task with 1 children and 0.73 gluten factor 
; constant factor of 1.5
; random seed: 691596

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child171 - child
    bread434 - bread-portion
    content427 - content-portion
    tray302 - tray
    table499 table317 table165 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray302 kitchen)
     (at_kitchen_bread bread434)
     (at_kitchen_content content427)
     (not_allergic_gluten child171)
     (waiting child171 table499)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child171)
    )
  )
)
