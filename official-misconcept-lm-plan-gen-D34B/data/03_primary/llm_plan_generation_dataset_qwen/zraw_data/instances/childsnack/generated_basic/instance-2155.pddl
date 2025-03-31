; child-snack task with 2 children and 0.53 gluten factor 
; constant factor of 1.9
; random seed: 366670

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child145 child91 - child
    bread228 bread144 - bread-portion
    content95 content145 - content-portion
    tray24 tray253 tray390 - tray
    table137 table398 table302 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray24 kitchen)
     (at tray253 kitchen)
     (at tray390 kitchen)
     (at_kitchen_bread bread228)
     (at_kitchen_bread bread144)
     (at_kitchen_content content95)
     (at_kitchen_content content145)
     (no_gluten_bread bread144)
     (no_gluten_content content95)
     (allergic_gluten child91)
     (not_allergic_gluten child145)
     (waiting child145 table302)
     (waiting child91 table398)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child145)
     (served child91)
    )
  )
)
