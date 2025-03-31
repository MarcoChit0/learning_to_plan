; child-snack task with 2 children and 0.41 gluten factor 
; constant factor of 1.3
; random seed: 299834

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child194 child192 - child
    bread438 bread8 - bread-portion
    content488 content79 - content-portion
    tray335 tray147 tray387 - tray
    table2 table457 table326 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray335 kitchen)
     (at tray147 kitchen)
     (at tray387 kitchen)
     (at_kitchen_bread bread438)
     (at_kitchen_bread bread8)
     (at_kitchen_content content488)
     (at_kitchen_content content79)
     (not_allergic_gluten child194)
     (not_allergic_gluten child192)
     (waiting child194 table326)
     (waiting child192 table326)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child194)
     (served child192)
    )
  )
)
