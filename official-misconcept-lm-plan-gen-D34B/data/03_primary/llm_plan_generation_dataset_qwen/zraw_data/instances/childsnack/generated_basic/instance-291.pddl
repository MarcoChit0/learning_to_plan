; child-snack task with 1 children and 0.75 gluten factor 
; constant factor of 1.7
; random seed: 622010

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child8 - child
    bread105 - bread-portion
    content55 - content-portion
    tray116 - tray
    table222 table298 table95 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray116 kitchen)
     (at_kitchen_bread bread105)
     (at_kitchen_content content55)
     (not_allergic_gluten child8)
     (waiting child8 table222)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child8)
    )
  )
)
