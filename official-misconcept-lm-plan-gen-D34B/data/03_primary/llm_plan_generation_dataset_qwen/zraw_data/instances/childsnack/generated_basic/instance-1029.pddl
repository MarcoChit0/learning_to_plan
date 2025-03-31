; child-snack task with 2 children and 0.41 gluten factor 
; constant factor of 1.3
; random seed: 24304

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child372 child210 - child
    bread39 bread429 - bread-portion
    content227 content259 - content-portion
    tray497 tray269 tray279 - tray
    table220 table124 table400 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray497 kitchen)
     (at tray269 kitchen)
     (at tray279 kitchen)
     (at_kitchen_bread bread39)
     (at_kitchen_bread bread429)
     (at_kitchen_content content227)
     (at_kitchen_content content259)
     (not_allergic_gluten child210)
     (not_allergic_gluten child372)
     (waiting child372 table124)
     (waiting child210 table220)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child372)
     (served child210)
    )
  )
)
