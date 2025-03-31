; child-snack task with 2 children and 0.58 gluten factor 
; constant factor of 1.5
; random seed: 899631

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child109 child138 - child
    bread233 bread209 - bread-portion
    content304 content51 - content-portion
    tray393 tray3 tray211 tray373 - tray
    table100 table278 table283 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray393 kitchen)
     (at tray3 kitchen)
     (at tray211 kitchen)
     (at tray373 kitchen)
     (at_kitchen_bread bread233)
     (at_kitchen_bread bread209)
     (at_kitchen_content content304)
     (at_kitchen_content content51)
     (no_gluten_bread bread233)
     (no_gluten_content content304)
     (allergic_gluten child138)
     (not_allergic_gluten child109)
     (waiting child109 table283)
     (waiting child138 table278)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child109)
     (served child138)
    )
  )
)
