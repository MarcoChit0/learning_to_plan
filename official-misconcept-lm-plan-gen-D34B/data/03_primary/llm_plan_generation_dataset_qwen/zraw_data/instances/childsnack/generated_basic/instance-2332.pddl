; child-snack task with 1 children and 0.7 gluten factor 
; constant factor of 1.3
; random seed: 662094

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child4 - child
    bread410 - bread-portion
    content64 - content-portion
    tray200 - tray
    table467 table372 table12 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray200 kitchen)
     (at_kitchen_bread bread410)
     (at_kitchen_content content64)
     (not_allergic_gluten child4)
     (waiting child4 table467)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child4)
    )
  )
)
