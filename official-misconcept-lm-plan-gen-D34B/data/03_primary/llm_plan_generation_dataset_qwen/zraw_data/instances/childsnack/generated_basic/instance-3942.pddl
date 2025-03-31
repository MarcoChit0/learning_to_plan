; child-snack task with 2 children and 0.78 gluten factor 
; constant factor of 1.9
; random seed: 471491

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child64 child169 - child
    bread226 bread208 - bread-portion
    content231 content472 - content-portion
    tray486 - tray
    table395 table165 table228 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray486 kitchen)
     (at_kitchen_bread bread226)
     (at_kitchen_bread bread208)
     (at_kitchen_content content231)
     (at_kitchen_content content472)
     (no_gluten_bread bread226)
     (no_gluten_content content472)
     (allergic_gluten child64)
     (not_allergic_gluten child169)
     (waiting child64 table228)
     (waiting child169 table395)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child64)
     (served child169)
    )
  )
)
