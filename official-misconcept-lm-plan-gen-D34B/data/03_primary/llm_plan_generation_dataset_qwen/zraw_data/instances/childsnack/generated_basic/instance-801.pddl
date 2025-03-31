; child-snack task with 1 children and 0.49 gluten factor 
; constant factor of 1.7
; random seed: 885582

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child202 - child
    bread400 - bread-portion
    content149 - content-portion
    tray198 tray154 - tray
    table333 table384 table478 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray198 kitchen)
     (at tray154 kitchen)
     (at_kitchen_bread bread400)
     (at_kitchen_content content149)
     (not_allergic_gluten child202)
     (waiting child202 table333)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child202)
    )
  )
)
