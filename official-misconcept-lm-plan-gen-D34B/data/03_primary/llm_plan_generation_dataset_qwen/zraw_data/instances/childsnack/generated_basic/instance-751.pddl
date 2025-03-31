; child-snack task with 2 children and 0.64 gluten factor 
; constant factor of 1.9
; random seed: 847815

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child194 child191 - child
    bread144 bread49 - bread-portion
    content292 content330 - content-portion
    tray343 tray205 - tray
    table329 table439 table341 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray343 kitchen)
     (at tray205 kitchen)
     (at_kitchen_bread bread144)
     (at_kitchen_bread bread49)
     (at_kitchen_content content292)
     (at_kitchen_content content330)
     (no_gluten_bread bread49)
     (no_gluten_content content292)
     (allergic_gluten child191)
     (not_allergic_gluten child194)
     (waiting child194 table341)
     (waiting child191 table341)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child194)
     (served child191)
    )
  )
)
