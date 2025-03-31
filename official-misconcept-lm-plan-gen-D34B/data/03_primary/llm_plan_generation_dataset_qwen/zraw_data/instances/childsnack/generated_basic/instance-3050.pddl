; child-snack task with 2 children and 0.58 gluten factor 
; constant factor of 1.5
; random seed: 251237

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child60 child262 - child
    bread485 bread356 - bread-portion
    content370 content232 - content-portion
    tray167 tray161 tray452 - tray
    table451 table362 table422 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray167 kitchen)
     (at tray161 kitchen)
     (at tray452 kitchen)
     (at_kitchen_bread bread485)
     (at_kitchen_bread bread356)
     (at_kitchen_content content370)
     (at_kitchen_content content232)
     (no_gluten_bread bread485)
     (no_gluten_content content370)
     (allergic_gluten child60)
     (not_allergic_gluten child262)
     (waiting child60 table451)
     (waiting child262 table451)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child60)
     (served child262)
    )
  )
)
