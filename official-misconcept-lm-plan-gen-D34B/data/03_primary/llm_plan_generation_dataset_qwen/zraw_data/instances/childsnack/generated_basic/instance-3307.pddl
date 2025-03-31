; child-snack task with 2 children and 0.58 gluten factor 
; constant factor of 1.5
; random seed: 888787

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child215 child449 - child
    bread274 bread113 - bread-portion
    content383 content220 - content-portion
    tray499 tray162 tray200 tray23 - tray
    table259 table313 table347 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray499 kitchen)
     (at tray162 kitchen)
     (at tray200 kitchen)
     (at tray23 kitchen)
     (at_kitchen_bread bread274)
     (at_kitchen_bread bread113)
     (at_kitchen_content content383)
     (at_kitchen_content content220)
     (no_gluten_bread bread274)
     (no_gluten_content content383)
     (allergic_gluten child449)
     (not_allergic_gluten child215)
     (waiting child215 table313)
     (waiting child449 table259)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child215)
     (served child449)
    )
  )
)
