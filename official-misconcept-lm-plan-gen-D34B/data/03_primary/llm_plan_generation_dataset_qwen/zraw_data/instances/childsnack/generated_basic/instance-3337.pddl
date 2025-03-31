; child-snack task with 2 children and 0.71 gluten factor 
; constant factor of 1.7
; random seed: 6396

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child339 child498 - child
    bread137 bread374 - bread-portion
    content423 content39 - content-portion
    tray157 tray235 tray462 tray210 - tray
    table72 table41 table178 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray157 kitchen)
     (at tray235 kitchen)
     (at tray462 kitchen)
     (at tray210 kitchen)
     (at_kitchen_bread bread137)
     (at_kitchen_bread bread374)
     (at_kitchen_content content423)
     (at_kitchen_content content39)
     (no_gluten_bread bread137)
     (no_gluten_content content423)
     (allergic_gluten child339)
     (not_allergic_gluten child498)
     (waiting child339 table178)
     (waiting child498 table41)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child339)
     (served child498)
    )
  )
)
