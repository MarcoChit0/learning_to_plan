; child-snack task with 2 children and 0.27 gluten factor 
; constant factor of 1.7
; random seed: 249922

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child484 child368 - child
    bread474 bread241 - bread-portion
    content167 content87 - content-portion
    tray195 tray472 tray110 - tray
    table427 table126 table392 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray195 kitchen)
     (at tray472 kitchen)
     (at tray110 kitchen)
     (at_kitchen_bread bread474)
     (at_kitchen_bread bread241)
     (at_kitchen_content content167)
     (at_kitchen_content content87)
     (not_allergic_gluten child484)
     (not_allergic_gluten child368)
     (waiting child484 table392)
     (waiting child368 table427)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child484)
     (served child368)
    )
  )
)
