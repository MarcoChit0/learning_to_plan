; child-snack task with 2 children and 0.63 gluten factor 
; constant factor of 1.3
; random seed: 338329

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child396 child416 - child
    bread357 bread340 - bread-portion
    content33 content367 - content-portion
    tray232 tray194 - tray
    table247 table376 table346 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray232 kitchen)
     (at tray194 kitchen)
     (at_kitchen_bread bread357)
     (at_kitchen_bread bread340)
     (at_kitchen_content content33)
     (at_kitchen_content content367)
     (no_gluten_bread bread340)
     (no_gluten_content content33)
     (allergic_gluten child416)
     (not_allergic_gluten child396)
     (waiting child396 table346)
     (waiting child416 table247)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child396)
     (served child416)
    )
  )
)
