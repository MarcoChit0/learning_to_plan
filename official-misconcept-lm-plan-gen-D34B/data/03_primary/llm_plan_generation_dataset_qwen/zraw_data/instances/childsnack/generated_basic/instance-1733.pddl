; child-snack task with 1 children and 0.36 gluten factor 
; constant factor of 1.3
; random seed: 143003

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child8 - child
    bread109 - bread-portion
    content388 - content-portion
    tray219 tray467 tray317 - tray
    table57 table240 table294 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray219 kitchen)
     (at tray467 kitchen)
     (at tray317 kitchen)
     (at_kitchen_bread bread109)
     (at_kitchen_content content388)
     (not_allergic_gluten child8)
     (waiting child8 table240)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child8)
    )
  )
)
