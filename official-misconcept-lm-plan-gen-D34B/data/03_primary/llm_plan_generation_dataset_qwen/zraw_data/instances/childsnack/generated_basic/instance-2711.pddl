; child-snack task with 3 children and 0.23 gluten factor 
; constant factor of 1.9
; random seed: 501689

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child103 child431 child55 - child
    bread248 bread451 bread215 - bread-portion
    content299 content334 content27 - content-portion
    tray398 tray323 tray443 - tray
    table377 table381 table427 - place
    sandw1 sandw2 sandw3 sandw4 sandw5 sandw6 - sandwich
  )
  (:init
     (at tray398 kitchen)
     (at tray323 kitchen)
     (at tray443 kitchen)
     (at_kitchen_bread bread248)
     (at_kitchen_bread bread451)
     (at_kitchen_bread bread215)
     (at_kitchen_content content299)
     (at_kitchen_content content334)
     (at_kitchen_content content27)
     (not_allergic_gluten child103)
     (not_allergic_gluten child431)
     (not_allergic_gluten child55)
     (waiting child103 table381)
     (waiting child431 table381)
     (waiting child55 table377)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
     (notexist sandw5)
     (notexist sandw6)
  )
  (:goal
    (and
     (served child103)
     (served child431)
     (served child55)
    )
  )
)
