; child-snack task with 3 children and 0.34 gluten factor 
; constant factor of 1.5
; random seed: 225573

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child428 child26 child288 - child
    bread457 bread282 bread389 - bread-portion
    content120 content219 content312 - content-portion
    tray200 tray356 tray28 - tray
    table24 table313 table206 - place
    sandw1 sandw2 sandw3 sandw4 sandw5 - sandwich
  )
  (:init
     (at tray200 kitchen)
     (at tray356 kitchen)
     (at tray28 kitchen)
     (at_kitchen_bread bread457)
     (at_kitchen_bread bread282)
     (at_kitchen_bread bread389)
     (at_kitchen_content content120)
     (at_kitchen_content content219)
     (at_kitchen_content content312)
     (no_gluten_bread bread282)
     (no_gluten_content content120)
     (allergic_gluten child288)
     (not_allergic_gluten child26)
     (not_allergic_gluten child428)
     (waiting child428 table24)
     (waiting child26 table24)
     (waiting child288 table313)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
     (notexist sandw5)
  )
  (:goal
    (and
     (served child428)
     (served child26)
     (served child288)
    )
  )
)
