; child-snack task with 2 children and 0.59 gluten factor 
; constant factor of 1.9
; random seed: 267512

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child490 child36 - child
    bread4 bread176 - bread-portion
    content66 content368 - content-portion
    tray413 tray496 - tray
    table72 table136 table269 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray413 kitchen)
     (at tray496 kitchen)
     (at_kitchen_bread bread4)
     (at_kitchen_bread bread176)
     (at_kitchen_content content66)
     (at_kitchen_content content368)
     (no_gluten_bread bread4)
     (no_gluten_content content66)
     (allergic_gluten child490)
     (not_allergic_gluten child36)
     (waiting child490 table136)
     (waiting child36 table269)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child490)
     (served child36)
    )
  )
)
