; child-snack task with 1 children and 0.7 gluten factor 
; constant factor of 1.3
; random seed: 787527

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child355 - child
    bread20 - bread-portion
    content440 - content-portion
    tray377 - tray
    table211 table353 table166 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray377 kitchen)
     (at_kitchen_bread bread20)
     (at_kitchen_content content440)
     (not_allergic_gluten child355)
     (waiting child355 table211)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child355)
    )
  )
)
