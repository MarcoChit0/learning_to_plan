; child-snack task with 2 children and 0.71 gluten factor 
; constant factor of 1.7
; random seed: 739989

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child52 child458 - child
    bread257 bread139 - bread-portion
    content314 content120 - content-portion
    tray67 tray52 tray324 tray478 - tray
    table44 table313 table399 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray67 kitchen)
     (at tray52 kitchen)
     (at tray324 kitchen)
     (at tray478 kitchen)
     (at_kitchen_bread bread257)
     (at_kitchen_bread bread139)
     (at_kitchen_content content314)
     (at_kitchen_content content120)
     (no_gluten_bread bread257)
     (no_gluten_content content120)
     (allergic_gluten child52)
     (not_allergic_gluten child458)
     (waiting child52 table399)
     (waiting child458 table44)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child52)
     (served child458)
    )
  )
)
