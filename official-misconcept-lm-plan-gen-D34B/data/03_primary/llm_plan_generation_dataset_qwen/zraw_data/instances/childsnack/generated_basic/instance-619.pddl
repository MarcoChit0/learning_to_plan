; child-snack task with 1 children and 0.63 gluten factor 
; constant factor of 1.5
; random seed: 646226

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child187 - child
    bread440 - bread-portion
    content254 - content-portion
    tray442 - tray
    table139 table49 table121 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray442 kitchen)
     (at_kitchen_bread bread440)
     (at_kitchen_content content254)
     (not_allergic_gluten child187)
     (waiting child187 table49)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child187)
    )
  )
)
