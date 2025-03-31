; child-snack task with 1 children and 0.23 gluten factor 
; constant factor of 1.7
; random seed: 962819

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child175 - child
    bread129 - bread-portion
    content139 - content-portion
    tray422 tray445 - tray
    table0 table222 table252 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray422 kitchen)
     (at tray445 kitchen)
     (at_kitchen_bread bread129)
     (at_kitchen_content content139)
     (not_allergic_gluten child175)
     (waiting child175 table222)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child175)
    )
  )
)
