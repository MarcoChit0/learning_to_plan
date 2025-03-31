; child-snack task with 1 children and 0.47 gluten factor 
; constant factor of 1.7
; random seed: 58470

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child172 - child
    bread243 - bread-portion
    content333 - content-portion
    tray128 tray250 tray494 tray33 - tray
    table256 table152 table107 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray128 kitchen)
     (at tray250 kitchen)
     (at tray494 kitchen)
     (at tray33 kitchen)
     (at_kitchen_bread bread243)
     (at_kitchen_content content333)
     (not_allergic_gluten child172)
     (waiting child172 table256)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child172)
    )
  )
)
