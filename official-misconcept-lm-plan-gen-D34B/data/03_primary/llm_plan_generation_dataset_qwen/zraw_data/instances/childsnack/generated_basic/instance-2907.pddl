; child-snack task with 2 children and 0.75 gluten factor 
; constant factor of 1.9
; random seed: 841362

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child326 child270 - child
    bread79 bread479 - bread-portion
    content126 content423 - content-portion
    tray449 tray279 tray68 - tray
    table47 table471 table268 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray449 kitchen)
     (at tray279 kitchen)
     (at tray68 kitchen)
     (at_kitchen_bread bread79)
     (at_kitchen_bread bread479)
     (at_kitchen_content content126)
     (at_kitchen_content content423)
     (no_gluten_bread bread79)
     (no_gluten_content content126)
     (allergic_gluten child270)
     (not_allergic_gluten child326)
     (waiting child326 table47)
     (waiting child270 table47)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child326)
     (served child270)
    )
  )
)
