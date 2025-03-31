; child-snack task with 2 children and 0.41 gluten factor 
; constant factor of 1.9
; random seed: 427692

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child347 child112 - child
    bread124 bread449 - bread-portion
    content292 content13 - content-portion
    tray75 tray101 tray296 tray280 - tray
    table134 table11 table335 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray75 kitchen)
     (at tray101 kitchen)
     (at tray296 kitchen)
     (at tray280 kitchen)
     (at_kitchen_bread bread124)
     (at_kitchen_bread bread449)
     (at_kitchen_content content292)
     (at_kitchen_content content13)
     (not_allergic_gluten child347)
     (not_allergic_gluten child112)
     (waiting child347 table335)
     (waiting child112 table335)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child347)
     (served child112)
    )
  )
)
