; child-snack task with 2 children and 0.75 gluten factor 
; constant factor of 1.7
; random seed: 907236

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child33 child433 - child
    bread38 bread116 - bread-portion
    content171 content421 - content-portion
    tray172 tray267 tray177 tray116 - tray
    table489 table218 table273 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray172 kitchen)
     (at tray267 kitchen)
     (at tray177 kitchen)
     (at tray116 kitchen)
     (at_kitchen_bread bread38)
     (at_kitchen_bread bread116)
     (at_kitchen_content content171)
     (at_kitchen_content content421)
     (no_gluten_bread bread38)
     (no_gluten_content content421)
     (allergic_gluten child33)
     (not_allergic_gluten child433)
     (waiting child33 table218)
     (waiting child433 table273)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child33)
     (served child433)
    )
  )
)
