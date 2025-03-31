; child-snack task with 2 children and 0.65 gluten factor 
; constant factor of 1.3
; random seed: 292452

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child490 child69 - child
    bread237 bread409 - bread-portion
    content55 content23 - content-portion
    tray427 tray142 tray462 tray410 - tray
    table107 table262 table197 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray427 kitchen)
     (at tray142 kitchen)
     (at tray462 kitchen)
     (at tray410 kitchen)
     (at_kitchen_bread bread237)
     (at_kitchen_bread bread409)
     (at_kitchen_content content55)
     (at_kitchen_content content23)
     (no_gluten_bread bread409)
     (no_gluten_content content55)
     (allergic_gluten child69)
     (not_allergic_gluten child490)
     (waiting child490 table107)
     (waiting child69 table197)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child490)
     (served child69)
    )
  )
)
