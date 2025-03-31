; child-snack task with 1 children and 0.45 gluten factor 
; constant factor of 1.9
; random seed: 946258

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child82 - child
    bread149 - bread-portion
    content417 - content-portion
    tray434 tray224 tray15 - tray
    table316 table409 table76 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray434 kitchen)
     (at tray224 kitchen)
     (at tray15 kitchen)
     (at_kitchen_bread bread149)
     (at_kitchen_content content417)
     (not_allergic_gluten child82)
     (waiting child82 table409)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child82)
    )
  )
)
