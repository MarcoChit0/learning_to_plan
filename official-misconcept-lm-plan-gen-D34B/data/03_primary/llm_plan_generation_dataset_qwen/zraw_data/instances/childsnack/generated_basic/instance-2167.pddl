; child-snack task with 1 children and 0.73 gluten factor 
; constant factor of 1.5
; random seed: 476935

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child71 - child
    bread116 - bread-portion
    content358 - content-portion
    tray408 - tray
    table270 table302 table492 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray408 kitchen)
     (at_kitchen_bread bread116)
     (at_kitchen_content content358)
     (not_allergic_gluten child71)
     (waiting child71 table302)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child71)
    )
  )
)
