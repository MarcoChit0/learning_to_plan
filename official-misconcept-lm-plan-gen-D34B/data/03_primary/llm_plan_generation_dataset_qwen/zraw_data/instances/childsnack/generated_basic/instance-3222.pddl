; child-snack task with 2 children and 0.55 gluten factor 
; constant factor of 1.9
; random seed: 701528

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child191 child367 - child
    bread85 bread236 - bread-portion
    content372 content88 - content-portion
    tray10 tray41 tray196 tray127 - tray
    table379 table341 table110 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray10 kitchen)
     (at tray41 kitchen)
     (at tray196 kitchen)
     (at tray127 kitchen)
     (at_kitchen_bread bread85)
     (at_kitchen_bread bread236)
     (at_kitchen_content content372)
     (at_kitchen_content content88)
     (no_gluten_bread bread85)
     (no_gluten_content content88)
     (allergic_gluten child367)
     (not_allergic_gluten child191)
     (waiting child191 table341)
     (waiting child367 table379)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child191)
     (served child367)
    )
  )
)
