; child-snack task with 1 children and 0.79 gluten factor 
; constant factor of 1.9
; random seed: 268545

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child205 - child
    bread246 - bread-portion
    content281 - content-portion
    tray263 tray130 tray403 - tray
    table386 table427 table376 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray263 kitchen)
     (at tray130 kitchen)
     (at tray403 kitchen)
     (at_kitchen_bread bread246)
     (at_kitchen_content content281)
     (not_allergic_gluten child205)
     (waiting child205 table376)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child205)
    )
  )
)
