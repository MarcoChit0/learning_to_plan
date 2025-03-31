; child-snack task with 2 children and 0.63 gluten factor 
; constant factor of 1.3
; random seed: 500035

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child459 child91 - child
    bread11 bread423 - bread-portion
    content97 content33 - content-portion
    tray54 tray373 - tray
    table311 table339 table141 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray54 kitchen)
     (at tray373 kitchen)
     (at_kitchen_bread bread11)
     (at_kitchen_bread bread423)
     (at_kitchen_content content97)
     (at_kitchen_content content33)
     (no_gluten_bread bread423)
     (no_gluten_content content97)
     (allergic_gluten child459)
     (not_allergic_gluten child91)
     (waiting child459 table311)
     (waiting child91 table339)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child459)
     (served child91)
    )
  )
)
