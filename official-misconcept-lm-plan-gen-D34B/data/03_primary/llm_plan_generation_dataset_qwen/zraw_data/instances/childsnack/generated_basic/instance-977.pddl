; child-snack task with 3 children and 0.33 gluten factor 
; constant factor of 1.7
; random seed: 431106

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child479 child383 child421 - child
    bread434 bread133 bread41 - bread-portion
    content305 content13 content14 - content-portion
    tray416 tray94 tray408 tray117 - tray
    table402 table19 table51 - place
    sandw1 sandw2 sandw3 sandw4 sandw5 sandw6 - sandwich
  )
  (:init
     (at tray416 kitchen)
     (at tray94 kitchen)
     (at tray408 kitchen)
     (at tray117 kitchen)
     (at_kitchen_bread bread434)
     (at_kitchen_bread bread133)
     (at_kitchen_bread bread41)
     (at_kitchen_content content305)
     (at_kitchen_content content13)
     (at_kitchen_content content14)
     (not_allergic_gluten child421)
     (not_allergic_gluten child383)
     (not_allergic_gluten child479)
     (waiting child479 table51)
     (waiting child383 table19)
     (waiting child421 table51)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
     (notexist sandw5)
     (notexist sandw6)
  )
  (:goal
    (and
     (served child479)
     (served child383)
     (served child421)
    )
  )
)
