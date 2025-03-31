; child-snack task with 1 children and 0.75 gluten factor 
; constant factor of 1.7
; random seed: 223068

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child285 - child
    bread436 - bread-portion
    content285 - content-portion
    tray109 - tray
    table397 table419 table274 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray109 kitchen)
     (at_kitchen_bread bread436)
     (at_kitchen_content content285)
     (not_allergic_gluten child285)
     (waiting child285 table419)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child285)
    )
  )
)
