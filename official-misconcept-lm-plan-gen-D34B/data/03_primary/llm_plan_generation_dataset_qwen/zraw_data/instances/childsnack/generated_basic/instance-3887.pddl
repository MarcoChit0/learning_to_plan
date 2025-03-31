; child-snack task with 1 children and 0.4 gluten factor 
; constant factor of 1.5
; random seed: 353444

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child367 - child
    bread27 - bread-portion
    content47 - content-portion
    tray464 tray432 - tray
    table14 table443 table285 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray464 kitchen)
     (at tray432 kitchen)
     (at_kitchen_bread bread27)
     (at_kitchen_content content47)
     (not_allergic_gluten child367)
     (waiting child367 table14)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child367)
    )
  )
)
