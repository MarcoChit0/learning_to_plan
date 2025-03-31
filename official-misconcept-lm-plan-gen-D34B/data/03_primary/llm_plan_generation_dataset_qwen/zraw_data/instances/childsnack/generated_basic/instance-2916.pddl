; child-snack task with 2 children and 0.75 gluten factor 
; constant factor of 1.9
; random seed: 895799

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child238 child416 - child
    bread391 bread276 - bread-portion
    content225 content341 - content-portion
    tray280 tray183 tray185 - tray
    table213 table412 table206 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray280 kitchen)
     (at tray183 kitchen)
     (at tray185 kitchen)
     (at_kitchen_bread bread391)
     (at_kitchen_bread bread276)
     (at_kitchen_content content225)
     (at_kitchen_content content341)
     (no_gluten_bread bread391)
     (no_gluten_content content341)
     (allergic_gluten child416)
     (not_allergic_gluten child238)
     (waiting child238 table206)
     (waiting child416 table412)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child238)
     (served child416)
    )
  )
)
