; child-snack task with 2 children and 0.55 gluten factor 
; constant factor of 1.7
; random seed: 873157

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child197 child359 - child
    bread293 bread294 - bread-portion
    content39 content209 - content-portion
    tray90 - tray
    table169 table385 table447 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray90 kitchen)
     (at_kitchen_bread bread293)
     (at_kitchen_bread bread294)
     (at_kitchen_content content39)
     (at_kitchen_content content209)
     (no_gluten_bread bread294)
     (no_gluten_content content39)
     (allergic_gluten child197)
     (not_allergic_gluten child359)
     (waiting child197 table385)
     (waiting child359 table385)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child197)
     (served child359)
    )
  )
)
