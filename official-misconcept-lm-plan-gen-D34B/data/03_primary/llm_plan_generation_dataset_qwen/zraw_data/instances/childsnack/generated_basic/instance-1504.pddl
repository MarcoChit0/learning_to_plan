; child-snack task with 2 children and 0.66 gluten factor 
; constant factor of 1.3
; random seed: 460357

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child318 child242 - child
    bread318 bread359 - bread-portion
    content363 content492 - content-portion
    tray55 tray5 tray382 - tray
    table357 table149 table355 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray55 kitchen)
     (at tray5 kitchen)
     (at tray382 kitchen)
     (at_kitchen_bread bread318)
     (at_kitchen_bread bread359)
     (at_kitchen_content content363)
     (at_kitchen_content content492)
     (no_gluten_bread bread318)
     (no_gluten_content content363)
     (allergic_gluten child242)
     (not_allergic_gluten child318)
     (waiting child318 table357)
     (waiting child242 table355)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child318)
     (served child242)
    )
  )
)
