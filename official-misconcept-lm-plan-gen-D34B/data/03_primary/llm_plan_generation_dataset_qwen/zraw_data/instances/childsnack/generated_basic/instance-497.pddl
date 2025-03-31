; child-snack task with 2 children and 0.32 gluten factor 
; constant factor of 1.9
; random seed: 456421

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child136 child142 - child
    bread352 bread24 - bread-portion
    content354 content396 - content-portion
    tray15 tray497 - tray
    table45 table261 table322 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray15 kitchen)
     (at tray497 kitchen)
     (at_kitchen_bread bread352)
     (at_kitchen_bread bread24)
     (at_kitchen_content content354)
     (at_kitchen_content content396)
     (not_allergic_gluten child136)
     (not_allergic_gluten child142)
     (waiting child136 table45)
     (waiting child142 table45)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child136)
     (served child142)
    )
  )
)
