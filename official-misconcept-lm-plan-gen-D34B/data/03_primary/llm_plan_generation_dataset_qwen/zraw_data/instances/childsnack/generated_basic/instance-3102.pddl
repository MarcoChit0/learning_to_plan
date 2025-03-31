; child-snack task with 2 children and 0.56 gluten factor 
; constant factor of 1.3
; random seed: 178640

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child81 child227 - child
    bread442 bread45 - bread-portion
    content352 content123 - content-portion
    tray153 tray205 - tray
    table443 table433 table260 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray153 kitchen)
     (at tray205 kitchen)
     (at_kitchen_bread bread442)
     (at_kitchen_bread bread45)
     (at_kitchen_content content352)
     (at_kitchen_content content123)
     (no_gluten_bread bread442)
     (no_gluten_content content123)
     (allergic_gluten child227)
     (not_allergic_gluten child81)
     (waiting child81 table260)
     (waiting child227 table433)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child81)
     (served child227)
    )
  )
)
