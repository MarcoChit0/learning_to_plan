; child-snack task with 1 children and 0.47 gluten factor 
; constant factor of 1.7
; random seed: 176245

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child18 - child
    bread243 - bread-portion
    content280 - content-portion
    tray455 tray349 tray124 tray252 - tray
    table217 table372 table460 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray455 kitchen)
     (at tray349 kitchen)
     (at tray124 kitchen)
     (at tray252 kitchen)
     (at_kitchen_bread bread243)
     (at_kitchen_content content280)
     (not_allergic_gluten child18)
     (waiting child18 table217)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child18)
    )
  )
)
