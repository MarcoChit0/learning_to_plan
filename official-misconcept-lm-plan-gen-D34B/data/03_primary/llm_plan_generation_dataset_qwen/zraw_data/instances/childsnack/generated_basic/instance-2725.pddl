; child-snack task with 3 children and 0.23 gluten factor 
; constant factor of 1.9
; random seed: 38471

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child449 child109 child248 - child
    bread428 bread63 bread125 - bread-portion
    content136 content289 content418 - content-portion
    tray51 tray119 tray258 - tray
    table465 table270 table464 - place
    sandw1 sandw2 sandw3 sandw4 sandw5 sandw6 - sandwich
  )
  (:init
     (at tray51 kitchen)
     (at tray119 kitchen)
     (at tray258 kitchen)
     (at_kitchen_bread bread428)
     (at_kitchen_bread bread63)
     (at_kitchen_bread bread125)
     (at_kitchen_content content136)
     (at_kitchen_content content289)
     (at_kitchen_content content418)
     (not_allergic_gluten child449)
     (not_allergic_gluten child248)
     (not_allergic_gluten child109)
     (waiting child449 table270)
     (waiting child109 table270)
     (waiting child248 table270)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
     (notexist sandw5)
     (notexist sandw6)
  )
  (:goal
    (and
     (served child449)
     (served child109)
     (served child248)
    )
  )
)
