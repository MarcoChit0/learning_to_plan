; child-snack task with 1 children and 0.73 gluten factor 
; constant factor of 1.5
; random seed: 798473

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child339 - child
    bread142 - bread-portion
    content468 - content-portion
    tray244 - tray
    table157 table454 table12 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray244 kitchen)
     (at_kitchen_bread bread142)
     (at_kitchen_content content468)
     (not_allergic_gluten child339)
     (waiting child339 table12)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child339)
    )
  )
)
