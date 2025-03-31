; child-snack task with 3 children and 0.33 gluten factor 
; constant factor of 1.7
; random seed: 685156

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child258 child181 child155 - child
    bread42 bread474 bread236 - bread-portion
    content422 content164 content247 - content-portion
    tray248 tray454 tray409 - tray
    table386 table255 table59 - place
    sandw1 sandw2 sandw3 sandw4 sandw5 sandw6 - sandwich
  )
  (:init
     (at tray248 kitchen)
     (at tray454 kitchen)
     (at tray409 kitchen)
     (at_kitchen_bread bread42)
     (at_kitchen_bread bread474)
     (at_kitchen_bread bread236)
     (at_kitchen_content content422)
     (at_kitchen_content content164)
     (at_kitchen_content content247)
     (not_allergic_gluten child181)
     (not_allergic_gluten child258)
     (not_allergic_gluten child155)
     (waiting child258 table386)
     (waiting child181 table386)
     (waiting child155 table255)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
     (notexist sandw5)
     (notexist sandw6)
  )
  (:goal
    (and
     (served child258)
     (served child181)
     (served child155)
    )
  )
)
