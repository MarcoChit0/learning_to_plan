; child-snack task with 1 children and 0.62 gluten factor 
; constant factor of 1.9
; random seed: 831382

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child225 - child
    bread111 - bread-portion
    content230 - content-portion
    tray122 tray67 tray484 - tray
    table411 table154 table136 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray122 kitchen)
     (at tray67 kitchen)
     (at tray484 kitchen)
     (at_kitchen_bread bread111)
     (at_kitchen_content content230)
     (not_allergic_gluten child225)
     (waiting child225 table154)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child225)
    )
  )
)
