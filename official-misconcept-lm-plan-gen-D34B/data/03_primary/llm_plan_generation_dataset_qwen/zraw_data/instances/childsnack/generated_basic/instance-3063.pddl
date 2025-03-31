; child-snack task with 2 children and 0.58 gluten factor 
; constant factor of 1.5
; random seed: 886520

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child382 child455 - child
    bread86 bread61 - bread-portion
    content198 content227 - content-portion
    tray346 tray228 tray393 - tray
    table214 table408 table61 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray346 kitchen)
     (at tray228 kitchen)
     (at tray393 kitchen)
     (at_kitchen_bread bread86)
     (at_kitchen_bread bread61)
     (at_kitchen_content content198)
     (at_kitchen_content content227)
     (no_gluten_bread bread61)
     (no_gluten_content content198)
     (allergic_gluten child455)
     (not_allergic_gluten child382)
     (waiting child382 table408)
     (waiting child455 table408)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child382)
     (served child455)
    )
  )
)
