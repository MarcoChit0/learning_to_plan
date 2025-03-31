; child-snack task with 2 children and 0.59 gluten factor 
; constant factor of 1.9
; random seed: 478200

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child115 child31 - child
    bread293 bread400 - bread-portion
    content114 content325 - content-portion
    tray443 tray276 - tray
    table63 table471 table218 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray443 kitchen)
     (at tray276 kitchen)
     (at_kitchen_bread bread293)
     (at_kitchen_bread bread400)
     (at_kitchen_content content114)
     (at_kitchen_content content325)
     (no_gluten_bread bread293)
     (no_gluten_content content325)
     (allergic_gluten child31)
     (not_allergic_gluten child115)
     (waiting child115 table471)
     (waiting child31 table63)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child115)
     (served child31)
    )
  )
)
