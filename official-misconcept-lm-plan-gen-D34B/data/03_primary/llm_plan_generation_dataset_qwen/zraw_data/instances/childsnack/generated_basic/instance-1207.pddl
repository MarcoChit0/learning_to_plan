; child-snack task with 3 children and 0.33 gluten factor 
; constant factor of 1.7
; random seed: 916691

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child283 child107 child260 - child
    bread392 bread296 bread486 - bread-portion
    content4 content196 content475 - content-portion
    tray249 tray269 tray493 - tray
    table147 table180 table48 - place
    sandw1 sandw2 sandw3 sandw4 sandw5 sandw6 - sandwich
  )
  (:init
     (at tray249 kitchen)
     (at tray269 kitchen)
     (at tray493 kitchen)
     (at_kitchen_bread bread392)
     (at_kitchen_bread bread296)
     (at_kitchen_bread bread486)
     (at_kitchen_content content4)
     (at_kitchen_content content196)
     (at_kitchen_content content475)
     (not_allergic_gluten child107)
     (not_allergic_gluten child283)
     (not_allergic_gluten child260)
     (waiting child283 table180)
     (waiting child107 table147)
     (waiting child260 table48)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
     (notexist sandw5)
     (notexist sandw6)
  )
  (:goal
    (and
     (served child283)
     (served child107)
     (served child260)
    )
  )
)
