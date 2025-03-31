; child-snack task with 2 children and 0.64 gluten factor 
; constant factor of 1.9
; random seed: 990494

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child495 child247 - child
    bread251 bread386 - bread-portion
    content295 content223 - content-portion
    tray146 tray451 - tray
    table96 table125 table356 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray146 kitchen)
     (at tray451 kitchen)
     (at_kitchen_bread bread251)
     (at_kitchen_bread bread386)
     (at_kitchen_content content295)
     (at_kitchen_content content223)
     (no_gluten_bread bread386)
     (no_gluten_content content223)
     (allergic_gluten child247)
     (not_allergic_gluten child495)
     (waiting child495 table356)
     (waiting child247 table356)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child495)
     (served child247)
    )
  )
)
