; child-snack task with 2 children and 0.58 gluten factor 
; constant factor of 1.5
; random seed: 681449

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child289 child201 - child
    bread202 bread446 - bread-portion
    content258 content439 - content-portion
    tray53 tray183 tray173 tray319 - tray
    table382 table91 table257 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray53 kitchen)
     (at tray183 kitchen)
     (at tray173 kitchen)
     (at tray319 kitchen)
     (at_kitchen_bread bread202)
     (at_kitchen_bread bread446)
     (at_kitchen_content content258)
     (at_kitchen_content content439)
     (no_gluten_bread bread202)
     (no_gluten_content content258)
     (allergic_gluten child289)
     (not_allergic_gluten child201)
     (waiting child289 table382)
     (waiting child201 table257)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child289)
     (served child201)
    )
  )
)
