; child-snack task with 1 children and 0.36 gluten factor 
; constant factor of 1.5
; random seed: 455596

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child466 - child
    bread214 - bread-portion
    content50 - content-portion
    tray19 - tray
    table225 table332 table141 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray19 kitchen)
     (at_kitchen_bread bread214)
     (at_kitchen_content content50)
     (not_allergic_gluten child466)
     (waiting child466 table141)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child466)
    )
  )
)
