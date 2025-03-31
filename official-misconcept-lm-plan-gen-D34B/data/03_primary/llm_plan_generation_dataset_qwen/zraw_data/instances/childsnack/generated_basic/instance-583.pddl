; child-snack task with 1 children and 0.63 gluten factor 
; constant factor of 1.5
; random seed: 172229

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child232 - child
    bread389 - bread-portion
    content330 - content-portion
    tray31 - tray
    table179 table209 table405 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray31 kitchen)
     (at_kitchen_bread bread389)
     (at_kitchen_content content330)
     (not_allergic_gluten child232)
     (waiting child232 table405)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child232)
    )
  )
)
