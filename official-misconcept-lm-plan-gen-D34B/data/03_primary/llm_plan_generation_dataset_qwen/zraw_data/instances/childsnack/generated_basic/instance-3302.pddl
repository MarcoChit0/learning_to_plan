; child-snack task with 2 children and 0.58 gluten factor 
; constant factor of 1.5
; random seed: 975107

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child305 child48 - child
    bread163 bread203 - bread-portion
    content224 content293 - content-portion
    tray46 tray115 tray29 tray328 - tray
    table306 table144 table293 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray46 kitchen)
     (at tray115 kitchen)
     (at tray29 kitchen)
     (at tray328 kitchen)
     (at_kitchen_bread bread163)
     (at_kitchen_bread bread203)
     (at_kitchen_content content224)
     (at_kitchen_content content293)
     (no_gluten_bread bread203)
     (no_gluten_content content224)
     (allergic_gluten child48)
     (not_allergic_gluten child305)
     (waiting child305 table306)
     (waiting child48 table144)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child305)
     (served child48)
    )
  )
)
