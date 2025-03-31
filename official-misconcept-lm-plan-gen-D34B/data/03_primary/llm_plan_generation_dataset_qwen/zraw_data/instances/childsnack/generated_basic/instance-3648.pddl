; child-snack task with 2 children and 0.79 gluten factor 
; constant factor of 1.9
; random seed: 463895

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child325 child89 - child
    bread18 bread220 - bread-portion
    content340 content144 - content-portion
    tray157 tray389 tray207 tray332 - tray
    table225 table476 table487 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray157 kitchen)
     (at tray389 kitchen)
     (at tray207 kitchen)
     (at tray332 kitchen)
     (at_kitchen_bread bread18)
     (at_kitchen_bread bread220)
     (at_kitchen_content content340)
     (at_kitchen_content content144)
     (no_gluten_bread bread18)
     (no_gluten_content content340)
     (allergic_gluten child325)
     (not_allergic_gluten child89)
     (waiting child325 table476)
     (waiting child89 table476)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child325)
     (served child89)
    )
  )
)
