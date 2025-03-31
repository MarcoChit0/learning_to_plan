; child-snack task with 1 children and 0.35 gluten factor 
; constant factor of 1.3
; random seed: 651079

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child478 - child
    bread128 - bread-portion
    content355 - content-portion
    tray392 tray68 tray96 tray73 - tray
    table486 table225 table163 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray392 kitchen)
     (at tray68 kitchen)
     (at tray96 kitchen)
     (at tray73 kitchen)
     (at_kitchen_bread bread128)
     (at_kitchen_content content355)
     (not_allergic_gluten child478)
     (waiting child478 table486)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child478)
    )
  )
)
