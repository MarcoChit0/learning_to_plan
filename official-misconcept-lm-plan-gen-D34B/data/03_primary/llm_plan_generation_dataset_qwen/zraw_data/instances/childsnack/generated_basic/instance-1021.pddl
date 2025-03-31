; child-snack task with 3 children and 0.3 gluten factor 
; constant factor of 1.3
; random seed: 625222

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child7 child120 child387 - child
    bread379 bread236 bread3 - bread-portion
    content17 content4 content287 - content-portion
    tray467 - tray
    table467 table210 table79 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray467 kitchen)
     (at_kitchen_bread bread379)
     (at_kitchen_bread bread236)
     (at_kitchen_bread bread3)
     (at_kitchen_content content17)
     (at_kitchen_content content4)
     (at_kitchen_content content287)
     (not_allergic_gluten child387)
     (not_allergic_gluten child120)
     (not_allergic_gluten child7)
     (waiting child7 table79)
     (waiting child120 table467)
     (waiting child387 table210)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child7)
     (served child120)
     (served child387)
    )
  )
)
