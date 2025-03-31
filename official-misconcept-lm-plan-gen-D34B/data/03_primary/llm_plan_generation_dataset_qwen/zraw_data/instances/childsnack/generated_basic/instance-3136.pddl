; child-snack task with 2 children and 0.62 gluten factor 
; constant factor of 1.3
; random seed: 135235

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child170 child294 - child
    bread284 bread52 - bread-portion
    content108 content147 - content-portion
    tray368 tray127 tray99 tray70 - tray
    table28 table122 table372 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray368 kitchen)
     (at tray127 kitchen)
     (at tray99 kitchen)
     (at tray70 kitchen)
     (at_kitchen_bread bread284)
     (at_kitchen_bread bread52)
     (at_kitchen_content content108)
     (at_kitchen_content content147)
     (no_gluten_bread bread52)
     (no_gluten_content content108)
     (allergic_gluten child170)
     (not_allergic_gluten child294)
     (waiting child170 table122)
     (waiting child294 table28)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child170)
     (served child294)
    )
  )
)
