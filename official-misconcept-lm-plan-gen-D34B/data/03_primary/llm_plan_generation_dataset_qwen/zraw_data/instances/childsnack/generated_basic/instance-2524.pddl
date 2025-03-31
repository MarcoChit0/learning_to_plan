; child-snack task with 1 children and 0.3 gluten factor 
; constant factor of 1.3
; random seed: 896995

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child200 - child
    bread466 - bread-portion
    content411 - content-portion
    tray343 tray107 tray372 - tray
    table204 table336 table153 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray343 kitchen)
     (at tray107 kitchen)
     (at tray372 kitchen)
     (at_kitchen_bread bread466)
     (at_kitchen_content content411)
     (not_allergic_gluten child200)
     (waiting child200 table153)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child200)
    )
  )
)
