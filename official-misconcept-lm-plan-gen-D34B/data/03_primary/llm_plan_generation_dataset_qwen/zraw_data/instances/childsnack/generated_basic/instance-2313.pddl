; child-snack task with 1 children and 0.7 gluten factor 
; constant factor of 1.3
; random seed: 495099

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child187 - child
    bread24 - bread-portion
    content13 - content-portion
    tray397 - tray
    table195 table49 table269 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray397 kitchen)
     (at_kitchen_bread bread24)
     (at_kitchen_content content13)
     (not_allergic_gluten child187)
     (waiting child187 table195)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child187)
    )
  )
)
