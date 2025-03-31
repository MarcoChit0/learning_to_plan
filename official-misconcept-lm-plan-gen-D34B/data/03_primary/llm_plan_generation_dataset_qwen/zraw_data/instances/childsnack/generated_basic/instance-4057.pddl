; child-snack task with 3 children and 0.22 gluten factor 
; constant factor of 1.9
; random seed: 371083

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child166 child274 child487 - child
    bread427 bread363 bread307 - bread-portion
    content284 content387 content487 - content-portion
    tray230 - tray
    table313 table86 table350 - place
    sandw1 sandw2 sandw3 sandw4 sandw5 sandw6 - sandwich
  )
  (:init
     (at tray230 kitchen)
     (at_kitchen_bread bread427)
     (at_kitchen_bread bread363)
     (at_kitchen_bread bread307)
     (at_kitchen_content content284)
     (at_kitchen_content content387)
     (at_kitchen_content content487)
     (not_allergic_gluten child487)
     (not_allergic_gluten child166)
     (not_allergic_gluten child274)
     (waiting child166 table350)
     (waiting child274 table350)
     (waiting child487 table86)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
     (notexist sandw5)
     (notexist sandw6)
  )
  (:goal
    (and
     (served child166)
     (served child274)
     (served child487)
    )
  )
)
