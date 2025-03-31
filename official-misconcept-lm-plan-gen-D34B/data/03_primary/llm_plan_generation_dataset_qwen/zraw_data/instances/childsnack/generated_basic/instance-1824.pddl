; child-snack task with 2 children and 0.58 gluten factor 
; constant factor of 1.9
; random seed: 37093

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child441 child16 - child
    bread255 bread246 - bread-portion
    content122 content284 - content-portion
    tray174 - tray
    table359 table250 table353 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray174 kitchen)
     (at_kitchen_bread bread255)
     (at_kitchen_bread bread246)
     (at_kitchen_content content122)
     (at_kitchen_content content284)
     (no_gluten_bread bread246)
     (no_gluten_content content122)
     (allergic_gluten child16)
     (not_allergic_gluten child441)
     (waiting child441 table250)
     (waiting child16 table359)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child441)
     (served child16)
    )
  )
)
