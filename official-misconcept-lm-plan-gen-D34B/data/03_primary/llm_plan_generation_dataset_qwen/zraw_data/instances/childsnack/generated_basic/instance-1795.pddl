; child-snack task with 1 children and 0.36 gluten factor 
; constant factor of 1.5
; random seed: 972673

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child239 - child
    bread416 - bread-portion
    content405 - content-portion
    tray128 - tray
    table415 table375 table162 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray128 kitchen)
     (at_kitchen_bread bread416)
     (at_kitchen_content content405)
     (not_allergic_gluten child239)
     (waiting child239 table415)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child239)
    )
  )
)
