; child-snack task with 1 children and 0.65 gluten factor 
; constant factor of 1.7
; random seed: 743358

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child438 - child
    bread456 - bread-portion
    content383 - content-portion
    tray420 tray463 tray162 - tray
    table463 table329 table259 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray420 kitchen)
     (at tray463 kitchen)
     (at tray162 kitchen)
     (at_kitchen_bread bread456)
     (at_kitchen_content content383)
     (not_allergic_gluten child438)
     (waiting child438 table329)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child438)
    )
  )
)
