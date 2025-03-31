; child-snack task with 2 children and 0.55 gluten factor 
; constant factor of 1.7
; random seed: 306262

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child244 child233 - child
    bread305 bread428 - bread-portion
    content466 content227 - content-portion
    tray211 - tray
    table188 table22 table273 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray211 kitchen)
     (at_kitchen_bread bread305)
     (at_kitchen_bread bread428)
     (at_kitchen_content content466)
     (at_kitchen_content content227)
     (no_gluten_bread bread428)
     (no_gluten_content content466)
     (allergic_gluten child244)
     (not_allergic_gluten child233)
     (waiting child244 table273)
     (waiting child233 table22)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child244)
     (served child233)
    )
  )
)
