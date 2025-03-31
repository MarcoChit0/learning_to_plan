; child-snack task with 1 children and 0.7 gluten factor 
; constant factor of 1.3
; random seed: 551051

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child179 - child
    bread177 - bread-portion
    content233 - content-portion
    tray316 - tray
    table168 table227 table195 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray316 kitchen)
     (at_kitchen_bread bread177)
     (at_kitchen_content content233)
     (not_allergic_gluten child179)
     (waiting child179 table227)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child179)
    )
  )
)
