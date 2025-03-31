; child-snack task with 1 children and 0.7 gluten factor 
; constant factor of 1.3
; random seed: 117891

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child75 - child
    bread102 - bread-portion
    content300 - content-portion
    tray229 - tray
    table113 table459 table375 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray229 kitchen)
     (at_kitchen_bread bread102)
     (at_kitchen_content content300)
     (not_allergic_gluten child75)
     (waiting child75 table459)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child75)
    )
  )
)
