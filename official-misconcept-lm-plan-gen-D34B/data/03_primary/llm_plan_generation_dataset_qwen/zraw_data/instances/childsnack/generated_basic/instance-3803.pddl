; child-snack task with 1 children and 0.46 gluten factor 
; constant factor of 1.9
; random seed: 779955

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child175 - child
    bread191 - bread-portion
    content116 - content-portion
    tray236 - tray
    table458 table138 table130 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray236 kitchen)
     (at_kitchen_bread bread191)
     (at_kitchen_content content116)
     (not_allergic_gluten child175)
     (waiting child175 table458)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child175)
    )
  )
)
