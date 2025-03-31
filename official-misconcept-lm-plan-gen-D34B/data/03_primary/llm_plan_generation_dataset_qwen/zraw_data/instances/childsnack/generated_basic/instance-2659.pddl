; child-snack task with 1 children and 0.35 gluten factor 
; constant factor of 1.3
; random seed: 826557

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child261 - child
    bread330 - bread-portion
    content109 - content-portion
    tray450 tray188 tray260 tray156 - tray
    table196 table319 table238 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray450 kitchen)
     (at tray188 kitchen)
     (at tray260 kitchen)
     (at tray156 kitchen)
     (at_kitchen_bread bread330)
     (at_kitchen_content content109)
     (not_allergic_gluten child261)
     (waiting child261 table196)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child261)
    )
  )
)
