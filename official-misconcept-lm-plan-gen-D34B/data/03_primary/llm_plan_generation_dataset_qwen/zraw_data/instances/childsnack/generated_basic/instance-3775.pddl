; child-snack task with 1 children and 0.46 gluten factor 
; constant factor of 1.9
; random seed: 542147

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child292 - child
    bread238 - bread-portion
    content159 - content-portion
    tray210 - tray
    table71 table411 table449 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray210 kitchen)
     (at_kitchen_bread bread238)
     (at_kitchen_content content159)
     (not_allergic_gluten child292)
     (waiting child292 table71)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child292)
    )
  )
)
