; child-snack task with 2 children and 0.79 gluten factor 
; constant factor of 1.9
; random seed: 702156

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child162 child347 - child
    bread88 bread365 - bread-portion
    content364 content32 - content-portion
    tray307 tray63 tray476 tray426 - tray
    table38 table123 table304 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray307 kitchen)
     (at tray63 kitchen)
     (at tray476 kitchen)
     (at tray426 kitchen)
     (at_kitchen_bread bread88)
     (at_kitchen_bread bread365)
     (at_kitchen_content content364)
     (at_kitchen_content content32)
     (no_gluten_bread bread365)
     (no_gluten_content content32)
     (allergic_gluten child162)
     (not_allergic_gluten child347)
     (waiting child162 table123)
     (waiting child347 table123)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child162)
     (served child347)
    )
  )
)
