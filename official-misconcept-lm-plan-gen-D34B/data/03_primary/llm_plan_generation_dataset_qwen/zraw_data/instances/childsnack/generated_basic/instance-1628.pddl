; child-snack task with 1 children and 0.71 gluten factor 
; constant factor of 1.3
; random seed: 620886

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child182 - child
    bread0 - bread-portion
    content367 - content-portion
    tray63 tray134 tray318 tray275 - tray
    table306 table9 table441 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray63 kitchen)
     (at tray134 kitchen)
     (at tray318 kitchen)
     (at tray275 kitchen)
     (at_kitchen_bread bread0)
     (at_kitchen_content content367)
     (not_allergic_gluten child182)
     (waiting child182 table306)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child182)
    )
  )
)
