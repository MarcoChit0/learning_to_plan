; child-snack task with 2 children and 0.58 gluten factor 
; constant factor of 1.5
; random seed: 564198

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child77 child61 - child
    bread464 bread120 - bread-portion
    content402 content20 - content-portion
    tray135 tray72 tray422 tray80 - tray
    table413 table361 table15 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray135 kitchen)
     (at tray72 kitchen)
     (at tray422 kitchen)
     (at tray80 kitchen)
     (at_kitchen_bread bread464)
     (at_kitchen_bread bread120)
     (at_kitchen_content content402)
     (at_kitchen_content content20)
     (no_gluten_bread bread120)
     (no_gluten_content content402)
     (allergic_gluten child77)
     (not_allergic_gluten child61)
     (waiting child77 table413)
     (waiting child61 table361)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child77)
     (served child61)
    )
  )
)
