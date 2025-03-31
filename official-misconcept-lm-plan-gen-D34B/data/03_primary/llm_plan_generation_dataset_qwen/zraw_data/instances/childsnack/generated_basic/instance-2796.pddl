; child-snack task with 2 children and 0.32 gluten factor 
; constant factor of 1.7
; random seed: 558990

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child314 child72 - child
    bread97 bread179 - bread-portion
    content243 content239 - content-portion
    tray398 tray62 - tray
    table409 table17 table222 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray398 kitchen)
     (at tray62 kitchen)
     (at_kitchen_bread bread97)
     (at_kitchen_bread bread179)
     (at_kitchen_content content243)
     (at_kitchen_content content239)
     (not_allergic_gluten child314)
     (not_allergic_gluten child72)
     (waiting child314 table222)
     (waiting child72 table17)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child314)
     (served child72)
    )
  )
)
