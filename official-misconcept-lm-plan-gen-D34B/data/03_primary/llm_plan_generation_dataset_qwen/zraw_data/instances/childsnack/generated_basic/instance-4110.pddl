; child-snack task with 1 children and 0.79 gluten factor 
; constant factor of 1.9
; random seed: 578520

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child84 - child
    bread463 - bread-portion
    content50 - content-portion
    tray0 tray467 tray5 - tray
    table375 table460 table358 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray0 kitchen)
     (at tray467 kitchen)
     (at tray5 kitchen)
     (at_kitchen_bread bread463)
     (at_kitchen_content content50)
     (not_allergic_gluten child84)
     (waiting child84 table460)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child84)
    )
  )
)
