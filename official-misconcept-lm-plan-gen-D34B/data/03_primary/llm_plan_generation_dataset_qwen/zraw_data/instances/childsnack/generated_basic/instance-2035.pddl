; child-snack task with 2 children and 0.34 gluten factor 
; constant factor of 1.5
; random seed: 517060

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child158 child260 - child
    bread138 bread94 - bread-portion
    content475 content406 - content-portion
    tray194 tray333 tray248 - tray
    table29 table144 table385 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray194 kitchen)
     (at tray333 kitchen)
     (at tray248 kitchen)
     (at_kitchen_bread bread138)
     (at_kitchen_bread bread94)
     (at_kitchen_content content475)
     (at_kitchen_content content406)
     (not_allergic_gluten child158)
     (not_allergic_gluten child260)
     (waiting child158 table385)
     (waiting child260 table144)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child158)
     (served child260)
    )
  )
)
