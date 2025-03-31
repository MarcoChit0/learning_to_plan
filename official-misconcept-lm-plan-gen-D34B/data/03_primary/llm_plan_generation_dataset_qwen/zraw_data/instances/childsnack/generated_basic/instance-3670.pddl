; child-snack task with 2 children and 0.8 gluten factor 
; constant factor of 1.7
; random seed: 948469

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child377 child444 - child
    bread90 bread91 - bread-portion
    content369 content480 - content-portion
    tray437 tray350 - tray
    table317 table399 table81 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray437 kitchen)
     (at tray350 kitchen)
     (at_kitchen_bread bread90)
     (at_kitchen_bread bread91)
     (at_kitchen_content content369)
     (at_kitchen_content content480)
     (no_gluten_bread bread91)
     (no_gluten_content content369)
     (allergic_gluten child444)
     (not_allergic_gluten child377)
     (waiting child377 table81)
     (waiting child444 table399)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child377)
     (served child444)
    )
  )
)
