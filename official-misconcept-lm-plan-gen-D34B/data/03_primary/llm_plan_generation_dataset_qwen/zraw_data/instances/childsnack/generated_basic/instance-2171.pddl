; child-snack task with 1 children and 0.73 gluten factor 
; constant factor of 1.5
; random seed: 958112

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child241 - child
    bread298 - bread-portion
    content412 - content-portion
    tray292 - tray
    table27 table358 table195 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray292 kitchen)
     (at_kitchen_bread bread298)
     (at_kitchen_content content412)
     (not_allergic_gluten child241)
     (waiting child241 table27)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child241)
    )
  )
)
