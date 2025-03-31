; child-snack task with 2 children and 0.59 gluten factor 
; constant factor of 1.9
; random seed: 664295

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child100 child320 - child
    bread396 bread41 - bread-portion
    content475 content214 - content-portion
    tray385 tray314 - tray
    table343 table16 table27 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray385 kitchen)
     (at tray314 kitchen)
     (at_kitchen_bread bread396)
     (at_kitchen_bread bread41)
     (at_kitchen_content content475)
     (at_kitchen_content content214)
     (no_gluten_bread bread41)
     (no_gluten_content content475)
     (allergic_gluten child100)
     (not_allergic_gluten child320)
     (waiting child100 table343)
     (waiting child320 table27)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child100)
     (served child320)
    )
  )
)
