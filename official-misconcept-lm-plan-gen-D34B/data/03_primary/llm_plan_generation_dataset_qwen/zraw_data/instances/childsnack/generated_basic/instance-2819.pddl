; child-snack task with 2 children and 0.32 gluten factor 
; constant factor of 1.7
; random seed: 366447

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child173 child428 - child
    bread50 bread362 - bread-portion
    content100 content477 - content-portion
    tray147 tray37 - tray
    table474 table181 table160 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray147 kitchen)
     (at tray37 kitchen)
     (at_kitchen_bread bread50)
     (at_kitchen_bread bread362)
     (at_kitchen_content content100)
     (at_kitchen_content content477)
     (not_allergic_gluten child428)
     (not_allergic_gluten child173)
     (waiting child173 table160)
     (waiting child428 table181)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child173)
     (served child428)
    )
  )
)
