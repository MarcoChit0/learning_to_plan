; child-snack task with 2 children and 0.58 gluten factor 
; constant factor of 1.5
; random seed: 264326

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child396 child167 - child
    bread446 bread422 - bread-portion
    content282 content190 - content-portion
    tray247 tray0 tray477 - tray
    table82 table462 table374 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray247 kitchen)
     (at tray0 kitchen)
     (at tray477 kitchen)
     (at_kitchen_bread bread446)
     (at_kitchen_bread bread422)
     (at_kitchen_content content282)
     (at_kitchen_content content190)
     (no_gluten_bread bread422)
     (no_gluten_content content190)
     (allergic_gluten child167)
     (not_allergic_gluten child396)
     (waiting child396 table462)
     (waiting child167 table462)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child396)
     (served child167)
    )
  )
)
