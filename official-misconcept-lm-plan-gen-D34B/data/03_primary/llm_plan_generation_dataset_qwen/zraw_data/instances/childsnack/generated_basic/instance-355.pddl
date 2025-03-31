; child-snack task with 1 children and 0.2 gluten factor 
; constant factor of 1.9
; random seed: 973435

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child323 - child
    bread320 - bread-portion
    content73 - content-portion
    tray184 tray33 tray75 tray108 - tray
    table67 table415 table26 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray184 kitchen)
     (at tray33 kitchen)
     (at tray75 kitchen)
     (at tray108 kitchen)
     (at_kitchen_bread bread320)
     (at_kitchen_content content73)
     (not_allergic_gluten child323)
     (waiting child323 table415)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child323)
    )
  )
)
