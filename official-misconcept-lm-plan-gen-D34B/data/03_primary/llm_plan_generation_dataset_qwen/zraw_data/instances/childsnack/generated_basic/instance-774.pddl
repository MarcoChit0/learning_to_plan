; child-snack task with 2 children and 0.64 gluten factor 
; constant factor of 1.9
; random seed: 328258

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child399 child148 - child
    bread220 bread50 - bread-portion
    content277 content333 - content-portion
    tray235 tray206 - tray
    table225 table481 table335 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray235 kitchen)
     (at tray206 kitchen)
     (at_kitchen_bread bread220)
     (at_kitchen_bread bread50)
     (at_kitchen_content content277)
     (at_kitchen_content content333)
     (no_gluten_bread bread50)
     (no_gluten_content content333)
     (allergic_gluten child148)
     (not_allergic_gluten child399)
     (waiting child399 table481)
     (waiting child148 table225)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child399)
     (served child148)
    )
  )
)
