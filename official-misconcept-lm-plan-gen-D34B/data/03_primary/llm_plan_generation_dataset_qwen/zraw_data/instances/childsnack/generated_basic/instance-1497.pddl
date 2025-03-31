; child-snack task with 2 children and 0.66 gluten factor 
; constant factor of 1.3
; random seed: 742174

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child413 child53 - child
    bread327 bread346 - bread-portion
    content249 content84 - content-portion
    tray422 tray239 tray18 - tray
    table159 table179 table390 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray422 kitchen)
     (at tray239 kitchen)
     (at tray18 kitchen)
     (at_kitchen_bread bread327)
     (at_kitchen_bread bread346)
     (at_kitchen_content content249)
     (at_kitchen_content content84)
     (no_gluten_bread bread327)
     (no_gluten_content content84)
     (allergic_gluten child53)
     (not_allergic_gluten child413)
     (waiting child413 table159)
     (waiting child53 table159)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child413)
     (served child53)
    )
  )
)
