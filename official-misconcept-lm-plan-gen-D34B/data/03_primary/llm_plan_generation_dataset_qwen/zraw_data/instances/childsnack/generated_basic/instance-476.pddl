; child-snack task with 2 children and 0.32 gluten factor 
; constant factor of 1.9
; random seed: 979098

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child143 child431 - child
    bread454 bread353 - bread-portion
    content272 content141 - content-portion
    tray230 tray449 - tray
    table488 table35 table143 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray230 kitchen)
     (at tray449 kitchen)
     (at_kitchen_bread bread454)
     (at_kitchen_bread bread353)
     (at_kitchen_content content272)
     (at_kitchen_content content141)
     (not_allergic_gluten child143)
     (not_allergic_gluten child431)
     (waiting child143 table35)
     (waiting child431 table488)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child143)
     (served child431)
    )
  )
)
