; child-snack task with 1 children and 0.77 gluten factor 
; constant factor of 1.9
; random seed: 115610

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child427 - child
    bread222 - bread-portion
    content316 - content-portion
    tray343 tray407 tray21 tray74 - tray
    table151 table307 table484 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray343 kitchen)
     (at tray407 kitchen)
     (at tray21 kitchen)
     (at tray74 kitchen)
     (at_kitchen_bread bread222)
     (at_kitchen_content content316)
     (not_allergic_gluten child427)
     (waiting child427 table484)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child427)
    )
  )
)
