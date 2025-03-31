; child-snack task with 1 children and 0.63 gluten factor 
; constant factor of 1.5
; random seed: 834061

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child49 - child
    bread81 - bread-portion
    content459 - content-portion
    tray117 - tray
    table73 table169 table388 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray117 kitchen)
     (at_kitchen_bread bread81)
     (at_kitchen_content content459)
     (not_allergic_gluten child49)
     (waiting child49 table388)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child49)
    )
  )
)
