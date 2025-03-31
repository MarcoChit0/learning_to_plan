; child-snack task with 2 children and 0.75 gluten factor 
; constant factor of 1.9
; random seed: 306669

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child15 child221 - child
    bread148 bread14 - bread-portion
    content209 content474 - content-portion
    tray234 tray109 tray384 - tray
    table8 table256 table413 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray234 kitchen)
     (at tray109 kitchen)
     (at tray384 kitchen)
     (at_kitchen_bread bread148)
     (at_kitchen_bread bread14)
     (at_kitchen_content content209)
     (at_kitchen_content content474)
     (no_gluten_bread bread14)
     (no_gluten_content content474)
     (allergic_gluten child15)
     (not_allergic_gluten child221)
     (waiting child15 table256)
     (waiting child221 table256)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child15)
     (served child221)
    )
  )
)
