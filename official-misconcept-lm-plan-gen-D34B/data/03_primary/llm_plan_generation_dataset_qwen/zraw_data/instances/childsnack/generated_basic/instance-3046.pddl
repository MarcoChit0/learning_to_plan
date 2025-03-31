; child-snack task with 2 children and 0.58 gluten factor 
; constant factor of 1.5
; random seed: 303826

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child456 child439 - child
    bread151 bread196 - bread-portion
    content306 content411 - content-portion
    tray223 tray339 tray222 - tray
    table53 table270 table377 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray223 kitchen)
     (at tray339 kitchen)
     (at tray222 kitchen)
     (at_kitchen_bread bread151)
     (at_kitchen_bread bread196)
     (at_kitchen_content content306)
     (at_kitchen_content content411)
     (no_gluten_bread bread196)
     (no_gluten_content content411)
     (allergic_gluten child439)
     (not_allergic_gluten child456)
     (waiting child456 table53)
     (waiting child439 table377)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child456)
     (served child439)
    )
  )
)
