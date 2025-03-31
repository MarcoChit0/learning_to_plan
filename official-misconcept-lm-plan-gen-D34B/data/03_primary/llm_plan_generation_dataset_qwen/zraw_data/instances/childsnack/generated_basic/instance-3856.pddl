; child-snack task with 1 children and 0.47 gluten factor 
; constant factor of 1.7
; random seed: 315267

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child293 - child
    bread471 - bread-portion
    content369 - content-portion
    tray271 tray498 tray360 tray366 - tray
    table182 table222 table479 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray271 kitchen)
     (at tray498 kitchen)
     (at tray360 kitchen)
     (at tray366 kitchen)
     (at_kitchen_bread bread471)
     (at_kitchen_content content369)
     (not_allergic_gluten child293)
     (waiting child293 table222)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child293)
    )
  )
)
