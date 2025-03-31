; child-snack task with 2 children and 0.75 gluten factor 
; constant factor of 1.9
; random seed: 729807

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child379 child369 - child
    bread383 bread172 - bread-portion
    content62 content5 - content-portion
    tray118 tray17 tray475 - tray
    table87 table154 table308 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray118 kitchen)
     (at tray17 kitchen)
     (at tray475 kitchen)
     (at_kitchen_bread bread383)
     (at_kitchen_bread bread172)
     (at_kitchen_content content62)
     (at_kitchen_content content5)
     (no_gluten_bread bread383)
     (no_gluten_content content62)
     (allergic_gluten child379)
     (not_allergic_gluten child369)
     (waiting child379 table308)
     (waiting child369 table87)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child379)
     (served child369)
    )
  )
)
